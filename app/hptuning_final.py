"""HPT final FNN: 1000 candidatos, Successive Halving y confirmación multisemilla.

Mantiene train/validación en GPU, conserva batch_size=262144 y no usa W4/test.
"""
from __future__ import annotations

import argparse, gc, json, math, os, shutil, time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from config import MODELS_DIR
from hptuning import (
    DOCS_HPT_DIR, GLOBAL_LOG_FILE, GPU_BATCHES, candidates, checkpoint,
    cpu_state, device_from, duration, evaluate, int_list, load_candidate,
    matrix, plot_progress, resolve_features, save_json, seed_all,
)
from train_fnn import FeatureScaler, PurchaseFNN


def validate_plan(count: int, epochs: list[int], survivors: list[int]) -> None:
    if len(epochs) != len(survivors):
        raise ValueError("--round-epochs y --survivors deben tener igual longitud")
    if any(b <= a for a, b in zip(epochs, epochs[1:])):
        raise ValueError("Las épocas deben crecer estrictamente")
    active = count
    for keep in survivors:
        if keep <= 0 or keep > active:
            raise ValueError("Plan de supervivientes inválido")
        active = keep


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize()


def train_to(model, optimizer, criterion, x, y, start_epoch, target_epoch,
             batch, seed, cid, label, progress_every=1):
    history, rows, device = [], int(x.shape[0]), x.device
    for epoch in range(start_epoch + 1, target_epoch + 1):
        epoch_seed = seed + cid * 100003 + epoch
        torch.manual_seed(epoch_seed)
        if device.type == "cuda": torch.cuda.manual_seed_all(epoch_seed)
        sync(device); t0 = time.time(); order = torch.randperm(rows, device=device)
        model.train(); total = 0.0; updates = 0
        for start in range(0, rows, batch):
            idx = order[start:start + batch]
            xb, yb = x.index_select(0, idx), y.index_select(0, idx)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
            total += float(loss.detach().item()) * int(idx.numel()); updates += 1
        sync(device); elapsed = time.time() - t0; train_loss = total / rows
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "duration_seconds": elapsed, "optimizer_updates": updates})
        if epoch == target_epoch or epoch == start_epoch + 1 or (epoch-start_epoch) % max(1, progress_every) == 0:
            print(f"\r[{label}][C{cid:04d}] época {epoch}/{target_epoch} | loss={train_loss:.6f} | updates={updates} | {duration(elapsed)}", end="", flush=True)
    if target_epoch > start_epoch: print(flush=True)
    return history


def plot_final_stability(rows: list[dict], path: Path) -> None:
    if not rows: return
    df = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)
    x = np.arange(len(df)); labels = [f"C{int(v):04d}" for v in df.candidate_id]
    fig, ax = plt.subplots(figsize=(9,5))
    ax.errorbar(x, df.mean_pr_auc, yerr=df.std_pr_auc, fmt="o", capsize=4)
    ax.set_xticks(x, labels); ax.set(xlabel="Finalista", ylabel="PR-AUC promedio ± DE", title="Estabilidad multisemilla")
    ax.grid(alpha=.25, axis="y"); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def confirm_finalists(ids, by_id, input_dim, xtr, ytr, xva, yva, yva_cpu,
                      epochs, n_seeds, base_seed, out, plots):
    final_dir = out / "final_confirmation"; final_dir.mkdir(parents=True, exist_ok=True)
    results, states = [], {}; total_runs = len(ids) * n_seeds; done = 0; started = time.time()
    print(f"\n[FINAL] {len(ids)} finalistas × {n_seeds} semillas × {epochs} épocas")
    for cid in ids:
        cfg = by_id[cid]
        for seed_idx in range(n_seeds):
            done += 1; run_seed = base_seed + 1_000_000 + cid * 1000 + seed_idx
            result_file = final_dir / f"candidate_{cid:04d}_seed_{seed_idx:02d}.json"
            model_file = final_dir / f"candidate_{cid:04d}_seed_{seed_idx:02d}.pth"
            if result_file.exists() and model_file.exists():
                rec = json.load(open(result_file, encoding="utf-8")); state = torch.load(model_file, map_location="cpu", weights_only=False)
            else:
                seed_all(run_seed)
                model = PurchaseFNN(input_dim, cfg["hidden_dims"], cfg["dropout"], cfg["activation"]).to(xtr.device)
                optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]], dtype=torch.float32, device=xtr.device))
                t0 = time.time()
                hist = train_to(model, optimizer, criterion, xtr, ytr, 0, epochs, cfg["batch_size"], run_seed, cid, f"FINAL-S{seed_idx}", max(1, epochs//10))
                metrics, _ = evaluate(model, criterion, xva, yva, yva_cpu, cfg["batch_size"])
                rec = {"candidate_id": cid, "seed_index": seed_idx, "seed": run_seed,
                       "epochs": epochs, "duration_seconds": time.time()-t0, **cfg, **metrics}
                state = cpu_state(model.state_dict()); torch.save(state, model_file)
                save_json(result_file, rec); save_json(final_dir / f"candidate_{cid:04d}_seed_{seed_idx:02d}_history.json", hist)
                del model, optimizer, criterion
                if xtr.device.type == "cuda": torch.cuda.empty_cache()
            results.append(rec); states[(cid, seed_idx)] = state
            pd.DataFrame(results).to_csv(final_dir / "final_seed_results.csv", index=False)
            eta = (time.time()-started)/done*(total_runs-done)
            print(f"[FINAL {done:02d}/{total_runs:02d}] C{cid:04d} seed={seed_idx} | PR-AUC={rec['pr_auc']:.6f} | F1={rec['f1']:.6f} | Lift@10={rec['lift_at_10']:.4f} | ETA={duration(eta)}")
    df = pd.DataFrame(results)
    summary = df.groupby("candidate_id", as_index=False).agg(
        mean_pr_auc=("pr_auc","mean"), std_pr_auc=("pr_auc","std"),
        min_pr_auc=("pr_auc","min"), max_pr_auc=("pr_auc","max"),
        mean_f1=("f1","mean"), mean_lift_at_10=("lift_at_10","mean"),
        mean_roc_auc=("roc_auc","mean"),
    ).fillna({"std_pr_auc":0.0}).sort_values(["mean_pr_auc","std_pr_auc","mean_lift_at_10"], ascending=[False,True,False])
    summary.to_csv(final_dir / "final_candidate_summary.csv", index=False)
    rows = summary.to_dict(orient="records"); save_json(final_dir / "final_candidate_summary.json", rows); plot_final_stability(rows, plots / "final_seed_stability.png")
    winner = int(summary.iloc[0].candidate_id)
    winner_runs = df[df.candidate_id == winner].sort_values("pr_auc", ascending=False)
    best_seed_idx = int(winner_runs.iloc[0].seed_index)
    confirmation = {"winner_candidate_id": winner, "winner_best_seed_index": best_seed_idx,
                    "winner_best_seed": int(winner_runs.iloc[0].seed),
                    "winner_aggregate_metrics": summary.iloc[0].to_dict(),
                    "winner_seed_metrics": winner_runs.to_dict(orient="records"),
                    "all_finalists": rows}
    save_json(out / "final_confirmation_summary.json", confirmation)
    return winner, by_id[winner], confirmation, states[(winner,best_seed_idx)]


def main() -> None:
    p = argparse.ArgumentParser(description="HPT final FNN con Successive Halving y confirmación multisemilla")
    p.add_argument("--train-parquet", required=True); p.add_argument("--feature-set", choices=["gain95","all"], required=True)
    p.add_argument("--experiment-name", default=None); p.add_argument("--candidates", type=int, default=1000)
    p.add_argument("--round-epochs", type=int_list, default=[5,15,40,80]); p.add_argument("--survivors", type=int_list, default=[200,40,8,5])
    p.add_argument("--final-seeds", type=int, default=3); p.add_argument("--final-epochs", type=int, default=120)
    p.add_argument("--val-size", type=float, default=.20); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-threads", type=int, default=min(32, os.cpu_count() or 1)); p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--max-rows", type=int, default=None); p.add_argument("--force", action="store_true"); args = p.parse_args()
    if not 0 < args.val_size < 1 or min(args.candidates,args.final_seeds,args.final_epochs) <= 0: raise ValueError("Parámetros inválidos")
    validate_plan(args.candidates, args.round_epochs, args.survivors)
    seed_all(args.seed); torch.set_num_threads(max(1,args.torch_threads)); device = device_from(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high"); torch.cuda.reset_peak_memory_stats()
    train_path = Path(args.train_parquet)
    if not train_path.exists(): raise FileNotFoundError(train_path)
    name = args.experiment_name or f"fnn_{args.feature_set}_final"
    out = MODELS_DIR / "hpt" / name; plots = DOCS_HPT_DIR / name; ckpts = out / "checkpoints"
    if args.force and out.exists(): shutil.rmtree(out)
    if args.force and plots.exists(): shutil.rmtree(plots)
    out.mkdir(parents=True, exist_ok=True); plots.mkdir(parents=True, exist_ok=True); ckpts.mkdir(parents=True, exist_ok=True)
    print(f"[HALVING] Experimento={name} | device={device} | plan={args.candidates}->{args.survivors} | épocas={args.round_epochs}")
    print(f"[HALVING] Confirmación={args.survivors[-1]}×{args.final_seeds} semillas×{args.final_epochs} épocas | batches={GPU_BATCHES if device.type=='cuda' else 'CPU'}")
    if device.type == "cuda": print(f"[HALVING] GPU: {torch.cuda.get_device_name(0)}")
    started=time.time(); df=pd.read_parquet(train_path)
    if "target" not in df or df.target.nunique()!=2: raise ValueError("target binario requerido")
    if args.max_rows and len(df)>args.max_rows:
        _,df=train_test_split(df,test_size=args.max_rows,random_state=args.seed,stratify=df.target); df=df.reset_index(drop=True)
    base,onehot,excluded=resolve_features(df,args.feature_set); raw,names=matrix(df,base,onehot); y=df.target.to_numpy(np.float32)
    itr,iva=train_test_split(np.arange(len(df)),test_size=args.val_size,random_state=args.seed,stratify=y)
    xtr0,xva0,ytr0,yva0=np.ascontiguousarray(raw[itr]),np.ascontiguousarray(raw[iva]),np.ascontiguousarray(y[itr]),np.ascontiguousarray(y[iva])
    scaler=FeatureScaler(); xtr0=scaler.fit_transform(xtr0,names).astype(np.float32,copy=False); xva0=scaler.transform(xva0,names).astype(np.float32,copy=False)
    del raw,itr,iva; gc.collect(); print(f"[HALVING] Moviendo tensores completos a {device}...")
    xtr,xva,ytr,yva=torch.from_numpy(xtr0).to(device),torch.from_numpy(xva0).to(device),torch.from_numpy(ytr0).to(device),torch.from_numpy(yva0).to(device)
    print(f"[HALVING] Filas={len(df):,} | train={len(ytr0):,} | val={len(yva0):,} | features={len(names)} | positivos={y.mean()*100:.2f}%")
    if device.type=="cuda": print(f"[HALVING] VRAM inicial: {torch.cuda.memory_allocated()/1024**2:,.1f} MiB")
    cfgs=candidates(args.candidates,args.seed,device); by_id={c['candidate_id']:c for c in cfgs}
    definition={"train_parquet":str(train_path.resolve()),"feature_set":args.feature_set,"candidate_count":args.candidates,
                "round_epochs":args.round_epochs,"survivors":args.survivors,"final_seeds":args.final_seeds,"final_epochs":args.final_epochs,
                "val_size":args.val_size,"seed":args.seed,"feature_columns":names,"gpu_batches":GPU_BATCHES,"candidates":cfgs}
    def_path=out/"run_definition.json"
    if def_path.exists() and not args.force:
        if json.load(open(def_path,encoding="utf-8"))!=definition: raise RuntimeError("Experimento existente incompatible; use otro nombre o --force")
        print("[HALVING] Reanudación detectada")
    else: save_json(def_path,definition)
    save_json(out/"candidate_configs.json",cfgs); save_json(out/"scaler.json",{"means":scaler.means,"stds":scaler.stds})
    active=list(by_id); all_records=[]
    for r,(target_epoch,keep) in enumerate(zip(args.round_epochs,args.survivors),1):
        r0=time.time(); records=[]; print(f"\n[HALVING] RONDA {r}/{len(args.round_epochs)} | candidatos={len(active)} | épocas={target_epoch} | pasan={keep}")
        for pos,cid in enumerate(active,1):
            t0=time.time(); cfg=by_id[cid]; path=ckpts/f"candidate_{cid:04d}.pt"
            model,opt,trained,hist,rmetrics=load_candidate(cfg,len(names),path,device,args.seed)
            criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['pos_weight']],dtype=torch.float32,device=device)); key=str(target_epoch)
            if trained<target_epoch:
                hist += train_to(model,opt,criterion,xtr,ytr,trained,target_epoch,cfg['batch_size'],args.seed,cid,f"HALVING-R{r}",max(1,(target_epoch-trained)//3)); trained=target_epoch
                result,_=evaluate(model,criterion,xva,yva,yva0,cfg['batch_size']); rmetrics[key]=result; checkpoint(path,cfg,model,opt,trained,hist,rmetrics)
            else: result=rmetrics[key] if key in rmetrics else evaluate(model,criterion,xva,yva,yva0,cfg['batch_size'])[0]
            elapsed=time.time()-t0; eta=(time.time()-r0)/pos*(len(active)-pos); steps=math.ceil(len(ytr0)/cfg['batch_size'])
            rec={"round":r,"target_epoch":target_epoch,"steps_per_epoch":steps,"updates_accumulated":steps*target_epoch,**cfg,**result,"duration_seconds":elapsed}
            records.append(rec); all_records.append(rec)
            pd.DataFrame(records).sort_values("pr_auc",ascending=False).to_csv(out/f"round_{r}_results.csv",index=False); pd.DataFrame(all_records).to_csv(out/"all_round_results.csv",index=False)
            print(f"[HALVING][R{r} {pos:04d}/{len(active):04d}] C{cid:04d} | PR-AUC={result['pr_auc']:.5f} | F1={result['f1']:.5f} | Lift@10={result['lift_at_10']:.3f} | batch={cfg['batch_size']:,} | updates={steps*target_epoch:,} | tiempo={duration(elapsed)} | ETA={duration(eta)}")
            del model,opt,criterion
            if device.type=="cuda": torch.cuda.empty_cache()
        ranked=sorted(records,key=lambda z:z['pr_auc'],reverse=True); active=[int(z['candidate_id']) for z in ranked[:keep]]; save_json(out/f"round_{r}_survivors.json",active)
        print(f"[HALVING] Supervivientes R{r}: {active} | duración={duration(time.time()-r0)}")
    winner,cfg,confirmation,state=confirm_finalists(active,by_id,len(names),xtr,ytr,xva,yva,yva0,args.final_epochs,args.final_seeds,args.seed,out,plots)
    torch.save(state,out/"best_model_state.pth"); save_json(out/"best_hparams.json",cfg)
    best_seed=int(confirmation['winner_best_seed']); seed_all(best_seed)
    model=PurchaseFNN(len(names),cfg['hidden_dims'],cfg['dropout'],cfg['activation']).to(device); model.load_state_dict(state)
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['pos_weight']],dtype=torch.float32,device=device)); best,proba=evaluate(model,criterion,xva,yva,yva0,cfg['batch_size'])
    summary={"timestamp":datetime.now().isoformat(),"experiment_name":name,"model":"FNN","search_method":"latin_hypercube_successive_halving_multiseed_confirmation",
             "objective":"maximize_mean_pr_auc_validation","feature_set":args.feature_set,"train_parquet":str(train_path),"n_rows_total":len(df),"n_rows_train":len(ytr0),"n_rows_validation":len(yva0),
             "target_positive_rate_total":float(y.mean()),"split":{"type":"row_stratified","validation_size":args.val_size,"seed":args.seed},"base_feature_columns":base,
             "model_feature_columns":names,"excluded_columns":excluded,"cycle_onehot":onehot,"candidate_count":args.candidates,"round_epochs":args.round_epochs,"survivors":args.survivors,
             "final_seeds":args.final_seeds,"final_epochs":args.final_epochs,"best_candidate_id":winner,"best_params":cfg,"best_seed":best_seed,
             "best_single_seed_metrics_validation":best,"confirmation":confirmation,"duration_seconds":round(time.time()-started,2),"device":str(device),
             "gpu_name":torch.cuda.get_device_name(0) if device.type=='cuda' else None,"torch_threads":torch.get_num_threads(),
             "max_gpu_memory_mib":round(torch.cuda.max_memory_allocated()/1024**2,2) if device.type=='cuda' else None}
    save_json(out/"best_metrics.json",summary); save_json(plots/"run_summary.json",summary); save_json(plots/"best_hparams.json",cfg)
    GLOBAL_LOG_FILE.parent.mkdir(parents=True,exist_ok=True)
    with open(GLOBAL_LOG_FILE,"a",encoding="utf-8") as f: f.write(json.dumps(summary,ensure_ascii=False)+"\n")
    plot_progress(all_records,plots/"halving_progress.png")
    pc,rc,_=precision_recall_curve(yva0,proba); fig,ax=plt.subplots(figsize=(7,6)); ax.plot(rc,pc); ax.axhline(yva0.mean(),linestyle="--"); ax.set(xlabel="Recall",ylabel="Precision",title="Curva Precision-Recall — mejor configuración confirmada"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(plots/"best_pr_curve.png",dpi=160,bbox_inches="tight"); plt.close(fig)
    agg=confirmation['winner_aggregate_metrics']
    print("\n[HALVING] ================================================================")
    print(f"[HALVING] Mejor candidato confirmado: C{winner:04d} | PR-AUC promedio={agg['mean_pr_auc']:.6f} ± {agg['std_pr_auc']:.6f}")
    print(f"[HALVING] Mejor semilla: {best_seed} | PR-AUC={best['pr_auc']:.6f} | F1={best['f1']:.6f} | Lift@10={best['lift_at_10']:.4f}")
    print(f"[HALVING] Parámetros: {cfg}"); print(f"[HALVING] Duración total: {duration(time.time()-started)}")
    if device.type=='cuda': print(f"[HALVING] Pico VRAM: {summary['max_gpu_memory_mib']:,.1f} MiB")
    print(f"[HALVING] Artefactos: {out}"); print(f"[HALVING] Resumen rastreable: {plots/'run_summary.json'}"); print("[HALVING] W4/test NO se utilizó durante el tuning.")
    print("[HALVING] ================================================================")


if __name__ == "__main__": main()
