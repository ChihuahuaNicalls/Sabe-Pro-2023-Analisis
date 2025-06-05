import pandas as pd
import os

def joinFiles(ruta_carpeta, columnas_comunes):
    todos_los_datos = pd.DataFrame()
    archivos_excel = [f for f in os.listdir(ruta_carpeta) if f.endswith(('.xls', '.xlsx'))]

    if not archivos_excel:
        print(f"No se encontraron archivos de Excel en la carpeta: {ruta_carpeta}")
        return

    print(f"Procesando {len(archivos_excel)} archivo(s) de Excel en '{ruta_carpeta}'...")

    for archivo in archivos_excel:
        ruta_completa_archivo = os.path.join(ruta_carpeta, archivo)
        try:
            df = pd.read_excel(ruta_completa_archivo)
            df.columns = df.columns.str.lower()

            columnas_comunes_lower = [col.lower() for col in columnas_comunes]

            columnas_a_extraer = [col for col in columnas_comunes_lower if col in df.columns]

            if not columnas_a_extraer:
                print(f"Advertencia: El archivo '{archivo}' no contiene ninguna de las columnas especificadas. Se ignorará.")
                continue

            df_seleccionado = df[columnas_a_extraer]

            todos_los_datos = pd.concat([todos_los_datos, df_seleccionado], ignore_index=True)
            print(f"  - '{archivo}' procesado correctamente.")

        except Exception as e:
            print(f"Error al procesar el archivo '{archivo}': {e}")

    if not todos_los_datos.empty:
        nombre_salida = "datos_combinados_filtrados.xlsx"
        ruta_salida = os.path.join(ruta_carpeta, nombre_salida)
        todos_los_datos.to_excel(ruta_salida, index=False)
        print(f"\n¡Proceso completado! Los datos combinados y filtrados se guardaron en: {ruta_salida}")
    else:
        print("\nNo se pudieron combinar datos de ningún archivo. Revisa los archivos y las columnas especificadas.")

#Columnas que se desean extraer de los archivos Excel
columnas_deseadas = [
    "ESTU_TIPODOCUMENTO", "ESTU_NACIONALIDAD", "ESTU_GENERO", "ESTU_FECHANACIMIENTO",
    "ESTU_EXTERIOR", "PERIODO", "ESTU_CONSECUTIVO", "ESTU_ESTUDIANTE",
    "ESTU_PAIS_RESIDE", "ESTU_DEPTO_RESIDE", "ESTU_COD_RESIDE_DEPTO",
    "ESTU_MCPIO_RESIDE", "ESTU_COD_RESIDE_MCPIO", "ESTU_AREARESIDE",
    "ESTU_TITULOOBTENIDOBACHILLER", "ESTU_VALORMATRICULAUNIVERSIDAD",
    "ESTU_PAGOMATRICULABECA", "ESTU_PAGOMATRICULACREDITO",
    "ESTU_PAGOMATRICULAPADRES", "ESTU_PAGOMATRICULAPROPIO",
    "ESTU_COMOCAPACITOEXAMENSB11", "ESTU_TIPODOCUMENTOSB11",
    "ESTU_SEMESTRECURSA", "FAMI_EDUCACIONPADRE", "FAMI_EDUCACIONMADRE",
    "FAMI_OCUPACIONPADRE", "FAMI_OCUPACIONMADRE", "FAMI_ESTRATOVIVIENDA",
    "FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR", "FAMI_TIENELAVADORA",
    "FAMI_TIENEHORNOMICROOGAS", "FAMI_TIENESERVICIOTV", "FAMI_TIENEAUTOMOVIL",
    "FAMI_TIENEMOTOCICLETA", "FAMI_TIENECONSOLAVIDEOJUEGOS",
    "FAMI_TRABAJOLABORPADRE", "FAMI_TRABAJOLABORMADRE",
    "ESTU_HORASSEMANATRABAJA", "ESTU_PAGOMATRICULA",
    "INST_COD_INSTITUCION", "INST_NOMBRE_INSTITUCION",
    "ESTU_PRGM_ACADEMICO", "ESTU_SNIES_PRGMACADEMICO", "GRUPOREFERENCIA",
    "ESTU_PRGM_CODMUNICIPIO", "ESTU_PRGM_MUNICIPIO", "ESTU_PRGM_DEPARTAMENTO",
    "ESTU_NIVEL_PRGM_ACADEMICO", "ESTU_METODO_PRGM", "ESTU_NUCLEO_PREGRADO",
    "ESTU_INST_CODMUNICIPIO", "ESTU_INST_MUNICIPIO", "ESTU_INST_DEPARTAMENTO",
    "INST_CARACTER_ACADEMICO", "INST_ORIGEN", "ESTU_PRIVADO_LIBERTAD",
    "ESTU_COD_MCPIO_PRESENTACION", "ESTU_MCPIO_PRESENTACION",
    "ESTU_DEPTO_PRESENTACION", "ESTU_COD_DEPTO_PRESENTACION",
    "MOD_RAZONA_CUANTITAT_PUNT", "MOD_RAZONA_CUANTITAT_DESEM",
    "MOD_RAZONA_CUANTITATIVO_PNAL", "MOD_RAZONA_CUANTITATIVO_PNBC",
    "MOD_LECTURA_CRITICA_PUNT", "MOD_LECTURA_CRITICA_DESEM",
    "MOD_LECTURA_CRITICA_PNAL", "MOD_LECTURA_CRITICA_PNBC",
    "MOD_COMPETEN_CIUDADA_PUNT", "MOD_COMPETEN_CIUDADA_DESEM",
    "MOD_COMPETEN_CIUDADA_PNAL", "MOD_COMPETEN_CIUDADA_PNBC",
    "MOD_INGLES_PUNT", "MOD_INGLES_DESEM", "MOD_INGLES_PNAL",
    "MOD_INGLES_PNBC", "MOD_COMUNI_ESCRITA_PUNT", "MOD_COMUNI_ESCRITA_DESEM",
    "MOD_COMUNI_ESCRITA_PNAL", "MOD_COMUNI_ESCRITA_PNBC",
    "PUNT_GLOBAL", "PERCENTIL_GLOBAL", "PERCENTIL_NBC", "ESTU_ESTADOINVESTIGACION"
]

if __name__ == "__main__":
    #Carpeta local
    carpeta_a_procesar = '.'
    joinFiles(carpeta_a_procesar, columnas_deseadas)