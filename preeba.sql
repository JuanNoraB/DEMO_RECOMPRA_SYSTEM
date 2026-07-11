create or replace PROCEDURE "PRC_UPDATE_PLAN_V2" (P_USER VARCHAR2)
as
  v_ln varchar2(18);
  v_id_col number;
  v_semana varchar2(36);
  v_semana_cerrada varchar2(36);
  v_semana_ant varchar2(36);
  v_semana_ant_2 varchar2(36);
  v_user varchar2(72);
  v_first_wd date;
  v_last_wd date;
  v_first_wd_1 date;
  v_last_wd_1 date;
  v_first_wd_2 date;
  v_last_wd_2 date;
  v_fecha date;
  v_existe_semana number;
  v_is_empty number;
  

begin

    -- get linea de negocio, id_semana 

      v_user := SUBSTR(P_USER,1,INSTR(P_USER,'@',1,1)-1);
      v_fecha := SYSDATE+12;

      SELECT NVL(LINEA_NEGOCIO,'NA') into v_ln
        FROM COLABORADOR
        WHERE USUARIO ||'@fpalatam.com' = P_USER; -- 'gabriel.troya@fpalatam.com';

        SELECT id_col into v_id_col
        FROM COLABORADOR
        WHERE USUARIO ||'@fpalatam.com' = P_USER; -- 'gabriel.troya@fpalatam.com';   
           

        -- SEMANA ACTUAL

              SELECT SEMANA_YYYY_WW, FIRST_DATE_WEEK, LAST_DATE_WEEK
              INTO v_semana, v_first_wd, v_last_wd
                FROM VW_MAT_FECHA
                WHERE FECHA = TRUNC(v_fecha,'IW') -- PARA PRUEBAS SYSDATE - 5
                ORDER BY 1;
        
        -- INSERTAR REGISTROS POR DEFECTO EN PLAN SEMANAL CUANDO NO HAY REGISTROS
        
           SELECT COUNT(1) INTO v_is_empty
           FROM plan_semanal
           WHERE ID_COL = v_id_col
            --AND ID_CLI||'-'||ID_PROYECTO IN ('1-6202','1-6203')
            ;
        
            IF v_is_empty = 0 THEN
            
                 INSERT INTO plan_semanal (
                    id_linea_negocio,
                    id_col,
                    id_semana,
                    id_cli,
                    id_proyecto,
                    descripcion,
                    semana_atras_2,
                    semana_atras_1,
                    semana_actual,
                    semana_adelante_1,
                    semana_adelante_2,
                    semana_cierre,
                    fecha_inicio_semana,
                    fecha_fin_semana
                )SELECT 
                    v_ln id_linea_negocio,
                    v_id_col id_col,
                    v_semana id_semana,
                    id_cli,
                    id_proyecto,
                    NULL descripcion,
                    semana_atras_2,
                    semana_atras_1,
                    semana_actual,
                    semana_adelante_1,
                    semana_adelante_2,
                    semana_cierre,
                    v_first_wd fecha_inicio_semana,
                    v_last_wd fecha_fin_semana
                   FROM PLAN_SEMANAL_DEFAULT; 
                COMMIT;
         END IF;
         
 -----------------------------------------
 
          -- SEMANA CERRADA PONER EXCEPTION
          --  v_semana_cerrada := v_semana; v_existe_semana
              
              SELECT COUNT(DISTINCT ID_SEMANA) INTO v_existe_semana
                 FROM PLAN_SEMANAL
                 WHERE ID_COL = v_id_col;  
           
            IF  v_existe_semana <> 0 THEN
            
               SELECT DISTINCT ID_SEMANA INTO v_semana_cerrada
               FROM PLAN_SEMANAL
               WHERE ID_COL = v_id_col;
            
            ELSE  
              
              v_semana_cerrada := v_semana;
            
            END IF;  
       
       DBMS_OUTPUT.put_line('v_semana_cerrada: '||v_semana_cerrada);
              DBMS_OUTPUT.put_line('v_semana: '||v_semana);
              
       --- LIMPIAR TABLAS TEMPORALES ---       
       
       DELETE FROM PLAN_SEMANAL_TMP 
        where id_col = v_id_col;
        commit;
        
        DELETE FROM PLAN_SEMANAL_TMP_100 
        where id_col = v_id_col;
        commit;
       
       DELETE FROM PLAN_SEMANAL_OLD 
        WHERE ID_COL = v_id_col;
        COMMIT; 
       
       -- SEMANA ANT 

             SELECT SEMANA_YYYY_WW, FIRST_DATE_WEEK, LAST_DATE_WEEK
              INTO v_semana_ant, v_first_wd_1, v_last_wd_1
                FROM VW_MAT_FECHA
                WHERE FECHA = TRUNC((v_fecha)-7,'IW') -- PARA PRUEBAS SYSDATE - 5
                ORDER BY 1;        

        -- SEMANA ANT 2 

             SELECT SEMANA_YYYY_WW, FIRST_DATE_WEEK, LAST_DATE_WEEK
              INTO v_semana_ant_2, v_first_wd_2, v_last_wd_2
                FROM VW_MAT_FECHA
                WHERE FECHA = TRUNC((v_fecha)-14,'IW') -- PARA PRUEBAS SYSDATE - 5
                ORDER BY 1;        

       
       
       IF  v_semana_cerrada = v_semana THEN
       
        
     --  LLENAR TABLA TEMPORAL PARA OPTIMIZAR

      INSERT INTO PLAN_SEMANAL_TMP  
      SELECT 'SEMANA - 2' AS SEMANA_ID,
        B.ID_COL,
          v_semana_ant_2 semana,
          B.ID_CLI,
          B.ID_PRO, 
          SUM(B.HORAS) AS SEMANA_VALOR
        FROM ACTIVIDADES_HORAS B
        WHERE b.id_col = v_id_col
        AND B.FECHA BETWEEN v_first_wd_2 AND  v_last_wd_2
        GROUP BY B.ID_COL,
          B.ID_CLI,
          B.ID_PRO
        UNION ALL
        SELECT 'SEMANA - 1' AS SEMANA_ID, 
         B.ID_COL,
          v_semana_ant semana,
          B.ID_CLI,
          B.ID_PRO, 
          SUM(B.HORAS) AS SEMANA_VALOR
        FROM ACTIVIDADES_HORAS B
        WHERE b.id_col = v_id_col
        AND B.FECHA BETWEEN v_first_wd_1 AND v_last_wd_1
        GROUP BY B.ID_COL,
          B.ID_CLI,
          B.ID_PRO
        UNION ALL
        SELECT 'SEMANA_ACTUAL' AS SEMANA_ID,
        B.ID_COL,
          v_semana semana,
          B.ID_CLI,
          B.ID_PRO, 
          SUM(B.HORAS) AS SEMANA_ACTUAL
        FROM ACTIVIDADES_HORAS B
        WHERE b.id_col = v_id_col
        AND B.FECHA BETWEEN v_first_wd AND v_last_wd
        GROUP BY B.ID_COL,
          B.ID_CLI,
          B.ID_PRO;
        COMMIT;
        
        --- SE CAMBIA "SIN PROYECTO" (2) POR OTROS (6203)
        
         UPDATE PLAN_SEMANAL_TMP SET 
         ID_PRO = 6203
         WHERE ID_COL = v_id_col
         AND ID_PRO = 2;
         COMMIT;
        
    -- MERGE DE DATOS PARA ACTUALIZAR O INSERTAR EN LA TABLA PLAN_SEMANAL 

       -- INSERT SEMANA - 2
        
        
            INSERT INTO PLAN_SEMANAL_TMP_100 
            (id_row,
             id_linea_negocio,
             id_col,
             id_semana,
             id_cli,
             id_proyecto,
             descripcion,
             semana_atras_2,
             semana_atras_1,
             semana_actual,
             semana_adelante_1,
             semana_adelante_2,
             semana_cierre,
             fecha_inicio_semana,
             fecha_fin_semana)
            SELECT SEQ_PLAN_SEMANAL_100.NEXTVAL,
            v_ln,
            ID_COL, 
            v_semana SEMANA,
            ID_CLI,
            ID_PRO,
            NULL,
            SEMANA_VALOR AS semana_atras_2,
            0,
             0,
             0,
             0,
             0,
             v_first_wd,
             v_last_wd
            FROM PLAN_SEMANAL_TMP
            WHERE ID_COL = v_id_col 
            AND SEMANA = v_semana_ant_2;



    -- INSERT SEMANA - 1

                  
         INSERT INTO PLAN_SEMANAL_TMP_100 
            (id_row,
             id_linea_negocio,
             id_col,
             id_semana,
             id_cli,
             id_proyecto,
             descripcion,
             semana_atras_2,
             semana_atras_1,
             semana_actual,
             semana_adelante_1,
             semana_adelante_2,
             semana_cierre,
             fecha_inicio_semana,
             fecha_fin_semana)
            SELECT SEQ_PLAN_SEMANAL_100.NEXTVAL,
            v_ln,
            ID_COL, 
            v_semana SEMANA,
            ID_CLI,
            ID_PRO,
            NULL,
            0,
            SEMANA_VALOR AS semana_atras_1,
             0,
             0,
             0,
             0,
             v_first_wd,
             v_last_wd
            FROM PLAN_SEMANAL_TMP
            WHERE ID_COL = v_id_col 
            AND SEMANA = v_semana_ant;    


    -- INSERT SEMANA ACTUAL
             
        INSERT INTO PLAN_SEMANAL_TMP_100 
            (id_row,
             id_linea_negocio,
             id_col,
             id_semana,
             id_cli,
             id_proyecto,
             descripcion,
             semana_atras_2,
             semana_atras_1,
             semana_actual,
             semana_adelante_1,
             semana_adelante_2,
             semana_cierre,
             fecha_inicio_semana,
             fecha_fin_semana)
            SELECT SEQ_PLAN_SEMANAL_100.NEXTVAL,
            v_ln,
            ID_COL, 
            v_semana SEMANA,
            ID_CLI,
            ID_PRO,
            NULL,
            0,
            0,
             SEMANA_VALOR AS semana_actual,
             0,
             0,
             0,
             v_first_wd,
             v_last_wd
            FROM PLAN_SEMANAL_TMP
            WHERE ID_COL = v_id_col 
            AND SEMANA = v_semana;     

        COMMIT;
        
        -- AGRUPACION DE VALORES
        
        -- GUARDAR DATOS ANTES DE BORRAR PARA PRESERVAR DESCRIPCIONES, SEMANA+1 Y SEMANA+2 
        
        
        INSERT INTO PLAN_SEMANAL_OLD
        SELECT * FROM PLAN_SEMANAL
        WHERE ID_COL = v_id_col;
        COMMIT;
        
        DELETE FROM PLAN_SEMANAL 
        WHERE ID_COL = v_id_col
        --AND ID_CLI||'-'||ID_PROYECTO NOT IN ('1-6202','1-6203')
        ;
        COMMIT;
        
        INSERT INTO PLAN_SEMANAL
        SELECT
            MAX(id_row),
            id_linea_negocio,
            id_col,
            id_semana,
            id_cli,
            id_proyecto,
            MAX(descripcion),
            SUM(semana_atras_2),
            SUM(semana_atras_1),
            SUM(semana_actual),
            SUM(semana_adelante_1),
            SUM(semana_adelante_2),
            SUM(semana_cierre),
            fecha_inicio_semana,
            fecha_fin_semana
        FROM
            plan_semanal_tmp_100
         WHERE ID_COL = v_id_col
        GROUP BY id_linea_negocio,
            id_col,
            id_semana,
            id_cli,
            id_proyecto,
            fecha_inicio_semana,
            fecha_fin_semana;
        COMMIT;
        
        -- ACTUALIZAR DESCRIPCIONES
        
        MERGE INTO PLAN_SEMANAL dest
        USING (
            SELECT ID_COL, 
            ID_SEMANA,
            ID_CLI,
            ID_PROYECTO,
            DESCRIPCION,
            semana_atras_2,
            semana_atras_1,
            semana_actual,
            semana_adelante_1,
            semana_adelante_2
            FROM PLAN_SEMANAL_OLD
            WHERE ID_COL = v_id_col 
        ) origen
        ON (dest.id_col = origen.id_col and 
            dest.id_semana = origen.ID_SEMANA and
            dest.id_cli = origen.id_cli and
            dest.id_proyecto = origen.ID_PROYECTO)        
        WHEN MATCHED THEN
            UPDATE SET
                dest.DESCRIPCION = origen.DESCRIPCION,
                dest.semana_adelante_1 = origen.semana_adelante_1,
                dest.semana_adelante_2 = origen.semana_adelante_2
        WHEN NOT MATCHED THEN
            INSERT (id_linea_negocio,
             id_col,
             id_semana,
             id_cli,
             id_proyecto,
             descripcion,
             semana_atras_2,
             semana_atras_1,
             semana_actual,
             semana_adelante_1,
             semana_adelante_2,
             semana_cierre,
             fecha_inicio_semana,
             fecha_fin_semana)
            VALUES (v_ln,
             origen.ID_COL,
             origen.ID_SEMANA,
             origen.ID_CLI,
             origen.ID_PROYECTO,
             origen.DESCRIPCION,
             origen.semana_atras_2,
             origen.semana_atras_1,
             origen.semana_actual,
             origen.semana_adelante_1,
             origen.semana_adelante_2,
             0,
             v_first_wd,
             v_last_wd
             );        
        COMMIT;        
     
        DELETE FROM PLAN_SEMANAL 
        WHERE ID_SEMANA = v_semana 
        and ID_COL = v_id_col
        AND semana_atras_2 = 0 AND
             semana_atras_1 = 0 AND
             semana_actual = 0 AND
             semana_adelante_1 = 0 AND
             semana_adelante_2 = 0 AND 
             ID_CLI||'-'||ID_PROYECTO NOT IN ('1-6202','1-6203');
            COMMIT;
     
        --- CIERRE POR ACTUALIZACION DE USUARIO --- 
       ELSE 
         -- INSERT DE FOTO DE DATOS A TABLA HISTÓRICA
        
            INSERT INTO plan_semanal_his (
            id_row,
            id_linea_negocio,
            id_col,
            id_semana,
            id_cli,
            id_proyecto,
            descripcion,
            semana_atras_2,
            semana_atras_1,
            semana_actual,
            semana_proyectada,
            semana_adelante_1,
            semana_adelante_2,
            semana_cierre,
            fecha_inicio_semana,
            fecha_fin_semana,
            fecha_guardado
          ) SELECT id_row,
            id_linea_negocio,
            id_col,
            v_semana as id_semana,
            id_cli,
            id_proyecto,
            descripcion,
            semana_atras_2,
            semana_atras_1,
            semana_actual,
            semana_adelante_1,
            semana_adelante_2,
            0,
            semana_actual semana_cierre,
            fecha_inicio_semana,
            fecha_fin_semana,
            SYSDATE  
          FROM PLAN_SEMANAL VT 
          WHERE ID_COL = v_id_col AND
          ID_SEMANA = v_semana_ant;
          COMMIT;
        
          DBMS_OUTPUT.put_line('INSERT EN HIS CON v_semana_ant: '||v_semana_ant);
        
            -- UPDATE DEL PASO DE SEMANA 
        
            UPDATE PLAN_SEMANAL SET 
            ID_SEMANA = v_semana,
            FECHA_INICIO_SEMANA = v_first_wd,
            FECHA_FIN_SEMANA = v_last_wd,
            SEMANA_ATRAS_2 = SEMANA_ATRAS_1,
            SEMANA_ATRAS_1 = SEMANA_ACTUAL,
            SEMANA_ACTUAL = SEMANA_ADELANTE_1,
            SEMANA_ADELANTE_1 = SEMANA_ADELANTE_2,
            SEMANA_ADELANTE_2 = 0,
            semana_cierre = 0
            WHERE 1=1;
        
            COMMIT;
        
            DELETE FROM PLAN_SEMANAL WHERE ID_SEMANA = v_semana_ant and ID_COL = v_id_col;
            COMMIT;
     
     END IF;
     
   -- 

end;
/