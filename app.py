import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sqlite3
import hashlib
import os
from pydriller import Repository
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

st.set_page_config(page_title="Sistema Predictivo de Defectos", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# Base de datos SQLite (analisis_repositorios, reportes_prioridad, credenciales)
# ─────────────────────────────────────────────────────────────────────────────
def inicializar_bd():
    conn = sqlite3.connect("sistema_predictivo.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analisis_repositorios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            repositorio TEXT NOT NULL,
            archivos_totales INTEGER NOT NULL,
            archivos_alto_riesgo INTEGER NOT NULL,
            porcentaje_riesgo REAL NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reportes_prioridad (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            repositorio TEXT NOT NULL,
            archivos_totales INTEGER NOT NULL,
            porcentaje_corte INTEGER NOT NULL,
            archivos_en_corte INTEGER NOT NULL,
            defectos_en_corte INTEGER NOT NULL,
            efectividad REAL NOT NULL
        )
    """)
    # Migracion automatica en caso de que la tabla reportes_prioridad ya existiera con un esquema anterior
    cursor.execute("PRAGMA table_info(reportes_prioridad)")
    columnas = [col[1] for col in cursor.fetchall()]
    if "repositorio" not in columnas:
        cursor.execute("ALTER TABLE reportes_prioridad ADD COLUMN repositorio TEXT DEFAULT ''")
    if "archivos_en_corte" not in columnas:
        cursor.execute("ALTER TABLE reportes_prioridad ADD COLUMN archivos_en_corte INTEGER DEFAULT 0")
    if "defectos_en_corte" not in columnas:
        cursor.execute("ALTER TABLE reportes_prioridad ADD COLUMN defectos_en_corte INTEGER DEFAULT 0")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS credenciales_admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario TEXT NOT NULL,
            hash_sha256 TEXT NOT NULL
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM credenciales_admin")
    if cursor.fetchone()[0] == 0:
        hash_default = hashlib.sha256("admin2026".encode()).hexdigest()
        cursor.execute(
            "INSERT INTO credenciales_admin (usuario, hash_sha256) VALUES (?, ?)",
            ("admin", hash_default)
        )
    conn.commit()
    conn.close()

inicializar_bd()

def guardar_analisis_bd(repo, total, alto, porcentaje):
    conn = sqlite3.connect("sistema_predictivo.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analisis_repositorios
        (repositorio, archivos_totales, archivos_alto_riesgo, porcentaje_riesgo)
        VALUES (?, ?, ?, ?)
    """, (repo, total, int(alto), round(porcentaje, 2)))
    conn.commit()
    conn.close()

def guardar_reporte_prioridad(repo, total, corte_pct, n_corte, defectos_en_corte, efectividad):
    conn = sqlite3.connect("sistema_predictivo.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO reportes_prioridad
        (repositorio, archivos_totales, porcentaje_corte, archivos_en_corte,
         defectos_en_corte, efectividad)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (repo, total, corte_pct, n_corte, defectos_en_corte, round(efectividad, 4)))
    conn.commit()
    conn.close()

def verificar_contrasena(contrasena_ingresada):
    hash_ingresado = hashlib.sha256(contrasena_ingresada.encode()).hexdigest()
    conn = sqlite3.connect("sistema_predictivo.db")
    cursor = conn.cursor()
    cursor.execute("SELECT hash_sha256 FROM credenciales_admin WHERE usuario = 'admin'")
    fila = cursor.fetchone()
    conn.close()
    if fila and fila[0] == hash_ingresado:
        return True
    return False

# ─────────────────────────────────────────────────────────────────────────────
# Carga y entrenamiento del modelo base (con cache)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_y_entrenar_modelo_base():
    df_maestro = pd.read_csv("dataset_maestro_final.csv").dropna()

    limite_loc = df_maestro['LOC'].quantile(0.99)
    limite_comp = df_maestro['Complejidad'].quantile(0.99)
    limite_churn = df_maestro['Churn'].quantile(0.99)

    df_limpio = df_maestro[
        (df_maestro['LOC'] <= limite_loc) &
        (df_maestro['Complejidad'] <= limite_comp) &
        (df_maestro['Churn'] <= limite_churn)
    ].copy()

    scaler = MinMaxScaler()
    df_limpio[['LOC', 'Complejidad', 'Churn']] = scaler.fit_transform(
        df_limpio[['LOC', 'Complejidad', 'Churn']]
    )

    X = df_limpio[['LOC', 'Complejidad', 'Churn']]
    y = df_limpio['Defectuoso']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)

    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    nb = GaussianNB()

    rf.fit(X_bal, y_bal)
    lr.fit(X_bal, y_bal)
    nb.fit(X_bal, y_bal)

    return rf, lr, nb, scaler, X_test, y_test

modelo_rf, modelo_lr, modelo_nb, escalador, X_test_global, y_test_global = cargar_y_entrenar_modelo_base()

# ─────────────────────────────────────────────────────────────────────────────
# Extraccion de metricas con PyDriller
# ─────────────────────────────────────────────────────────────────────────────
def analizar_repositorio_en_vivo(url_o_ruta, limite_commits=300):
    datos_archivos = {}
    contador = 0
    for commit in Repository(url_o_ruta).traverse_commits():
        contador += 1
        if contador > limite_commits:
            break
        for mod in commit.modified_files:
            if mod.filename and mod.filename.endswith('.java'):
                ruta = mod.new_path or mod.old_path or mod.filename
                if ruta not in datos_archivos:
                    datos_archivos[ruta] = {'loc': 0, 'complejidad': 0, 'churn': 0}
                datos_archivos[ruta]['churn'] += (mod.added_lines + mod.deleted_lines)
                if mod.nloc is not None:
                    datos_archivos[ruta]['loc'] = mod.nloc
                if mod.complexity is not None:
                    datos_archivos[ruta]['complejidad'] = mod.complexity
    filas = [
        {'Archivo': arch, 'LOC': m['loc'], 'Complejidad': m['complejidad'], 'Churn': m['churn']}
        for arch, m in datos_archivos.items()
    ]
    return pd.DataFrame(filas)

def calcular_diagnostico_y_probabilidades(df):
    """Calcula las probabilidades relativas de los tres tipos de defectos basados en sus metricas."""
    # Las tres metricas asociadas:
    # 1. Defectos de Regresion / Inestabilidad -> Churn
    # 2. Defectos Logicos y Algoritmicos -> Complejidad Ciclomatica (CC)
    # 3. Defectos de Rendimiento / Sobrecarga -> Lineas de Codigo (LOC)
    
    suma = df['LOC_norm'] + df['CC_norm'] + df['Churn_norm']
    
    # Manejar caso donde las tres metricas sean 0 (repartir equitativamente)
    df['Prob_Regresion'] = np.where(suma > 0, (df['Churn_norm'] / suma) * df['Probabilidad_Defecto'], 0.0)
    df['Prob_Logico'] = np.where(suma > 0, (df['CC_norm'] / suma) * df['Probabilidad_Defecto'], 0.0)
    df['Prob_Rendimiento'] = np.where(suma > 0, (df['LOC_norm'] / suma) * df['Probabilidad_Defecto'], 0.0)
    
    def obtener_diagnostico(row):
        p_reg = row['Prob_Regresion']
        p_log = row['Prob_Logico']
        p_ren = row['Prob_Rendimiento']
        max_p = max(p_reg, p_log, p_ren)
        if max_p == 0:
            return "Sin factores de riesgo detectados"
        if p_reg == max_p:
            return f"Regresion / Inestabilidad ({p_reg:.1%})"
        elif p_log == max_p:
            return f"Logico / Bifurcaciones ({p_log:.1%})"
        else:
            return f"Rendimiento / Tamano ({p_ren:.1%})"

    df['Diagnostico_Riesgo'] = df.apply(obtener_diagnostico, axis=1)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# DETECCION DE MODO ADMINISTRADOR (PARAMETRO OCULTO EN URL: ?admin=true)
# ─────────────────────────────────────────────────────────────────────────────
modo_admin_solicitado = False
parametros_url = st.query_params

if "admin" in parametros_url and parametros_url["admin"].lower() in ["true", "1", "si"]:
    modo_admin_solicitado = True

# ─────────────────────────────────────────────────────────────────────────────
# VISTA PRIVADA Y PROTEGIDA: ADMINISTRADOR DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────
if modo_admin_solicitado:
    st.sidebar.title("Consola del Administrador")
    st.sidebar.markdown("**Modo de Mantenimiento Tecnico**")
    st.sidebar.caption("Acceso privado mediante parametro seguro")
    st.sidebar.markdown("---")

    if st.sidebar.button("Volver al Modo Analista"):
        st.query_params.clear()
        st.session_state["admin_autenticado"] = False
        st.rerun()

    if "admin_autenticado" not in st.session_state:
        st.session_state["admin_autenticado"] = False

    if not st.session_state["admin_autenticado"]:
        st.header("Autenticacion Requerida")
        st.warning("Este panel esta reservado para el Administrador del Modelo. Por favor ingresa tus credenciales.")
        with st.form("form_login"):
            contrasena = st.text_input("Contrasena de Administrador", type="password")
            enviado = st.form_submit_button("Ingresar a la Consola")
        if enviado:
            if verificar_contrasena(contrasena):
                st.session_state["admin_autenticado"] = True
                st.rerun()
            else:
                st.error("Contrasena incorrecta. Acceso denegado.")
        st.stop()

    st.header("Panel del Administrador del Modelo")
    st.markdown("Consola tecnica para la auditoria de modelos, metricas cuantitativas y gobernanza de datos.")

    if st.sidebar.button("Cerrar Sesion"):
        st.session_state["admin_autenticado"] = False
        st.rerun()

    tab_bench, tab_roc, tab_corpus, tab_reportes = st.tabs([
        "Benchmarking de Modelos",
        "Curvas ROC Comparativas",
        "Corpus Maestro y Parametros",
        "Reportes de Efectividad"
    ])

    with tab_bench:
        st.subheader("Tabla Comparativa de Rendimiento de Algoritmos")
        st.caption("Evaluacion sobre el conjunto de prueba independiente del 20% (2,998 archivos reales).")

        modelos_eval = {
            "Random Forest": modelo_rf,
            "Regresion Logistica": modelo_lr,
            "Naive Bayes": modelo_nb
        }

        filas_bench = []
        for nombre, mod in modelos_eval.items():
            pred = mod.predict(X_test_global)
            prob = mod.predict_proba(X_test_global)[:, 1]
            filas_bench.append({
                "Modelo": nombre,
                "Accuracy": round(accuracy_score(y_test_global, pred), 2),
                "Precision (Clase 1)": round(precision_score(y_test_global, pred, zero_division=0), 2),
                "Recall (Clase 1)": round(recall_score(y_test_global, pred, zero_division=0), 2),
                "F1-Score (Clase 1)": round(f1_score(y_test_global, pred, zero_division=0), 2),
                "AUC-ROC": round(roc_auc_score(y_test_global, prob), 2)
            })

        df_bench = pd.DataFrame(filas_bench)
        st.dataframe(
            df_bench.style.highlight_max(
                subset=["Recall (Clase 1)", "F1-Score (Clase 1)", "AUC-ROC"],
                color="#d4edda"
            ),
            use_container_width=True
        )

        st.subheader("Matriz de Confusion - Random Forest")
        pred_rf = modelo_rf.predict(X_test_global)
        tn, fp, fn, tp = confusion_matrix(y_test_global, pred_rf).ravel()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Verdaderos Positivos (VP)", int(tp), help="Defectos reales detectados correctamente")
        m2.metric("Falsos Positivos (FP)", int(fp), help="Archivos sanos marcados como sospechosos")
        m3.metric("Verdaderos Negativos (VN)", int(tn), help="Archivos sanos clasificados correctamente")
        m4.metric("Falsos Negativos (FN)", int(fn), help="Defectos reales que escaparon al modelo")

        st.caption("El peor error en calidad es el Falso Negativo (FN), ya que deja escapar codigo con fallas a produccion.")

    with tab_roc:
        st.subheader("Curvas ROC Comparativas entre Modelos")
        # Buscar la imagen localmente en desarrollo o en la carpeta figuras
        roc_path_desarrollo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curva_roc_comparativa.png")
        roc_path_figuras = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figuras", "curva_roc_comparativa.png")

        roc_path = roc_path_desarrollo if os.path.exists(roc_path_desarrollo) else roc_path_figuras

        if os.path.exists(roc_path):
            img = mpimg.imread(roc_path)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.imshow(img)
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("Imagen de curvas ROC no encontrada.")

        st.markdown("""
        **Interpretacion tecnica del grafico:**
        - La curva azul de **Random Forest (AUC = 0.77)** domina sobre los modelos base en todos los umbrales.
        - La diagonal punteada (AUC = 0.50) representa una clasificacion aleatoria.
        - El AUC de 0.77 confirma la separabilidad efectiva de archivos con defecto frente a sanos.
        """)

    with tab_corpus:
        st.subheader("Corpus Maestro de Entrenamiento")
        st.write("Cinco repositorios de la Fundacion Apache analizados para la calibracion del modelo:")
        proyectos_apache = {
            "Apache Commons Lang": "Utilidades de procesamiento de cadenas y texto en Java",
            "Apache Commons IO": "Operaciones de lectura y escritura de archivos",
            "Apache Commons Math": "Algoritmos matematicos y estadisticos",
            "Apache Commons CLI": "Procesamiento de comandos e interfaces de linea",
            "Apache JSPWiki": "Aplicacion web completa de gestion de contenidos"
        }
        for nombre, desc in proyectos_apache.items():
            st.markdown(f"- **{nombre}** — {desc}")

        st.subheader("Parametros del Modelo Activo")
        params = {
            "Algoritmo principal": "Random Forest",
            "Cantidad de estimadores": "100 arboles de decision",
            "Criterio de division": "Impureza de Gini",
            "Tecnica de balanceo": "SMOTE (k_neighbors=5, random_state=42)",
            "Normalizacion": "MinMaxScaler [0, 1]",
            "Particion de datos": "80% entrenamiento / 20% prueba (random_state=42)",
            "Metricas de entrada": "LOC, Complejidad Ciclomatica, Churn Absoluto Acumulado",
            "Filtro de valores atipicos": "Percentil 99 por cada metrica",
            "Registros de entrenamiento": "11,989 archivos (7,090 sanos + 4,899 defectuosos)",
            "Registros tras SMOTE": "14,180 archivos balanceados (7,090 + 7,090)",
            "Registros de prueba": "2,998 archivos reales (1,760 sanos + 1,238 defectuosos)"
        }
        for k, v in params.items():
            st.markdown(f"- **{k}:** {v}")

    with tab_reportes:
        st.subheader("Reportes de Efectividad del Corte Prioritario")
        conn = sqlite3.connect("sistema_predictivo.db")
        df_rep = pd.read_sql_query(
            "SELECT id, fecha, repositorio, archivos_totales, porcentaje_corte, "
            "archivos_en_corte, defectos_en_corte, efectividad "
            "FROM reportes_prioridad ORDER BY id DESC",
            conn
        )
        conn.close()
        if not df_rep.empty:
            df_rep['efectividad'] = df_rep['efectividad'].map(lambda x: f"{x:.1%}")
            st.dataframe(df_rep, use_container_width=True)
        else:
            st.info("No hay reportes de efectividad todavia.")

# ─────────────────────────────────────────────────────────────────────────────
# VISTA PREDETERMINADA: ANALISTA DE CALIDAD (OPERATIVO Y PUBLICO)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.sidebar.title("Sistema Predictivo de Defectos")
    st.sidebar.markdown("**Modulo de Aseguramiento de Calidad**")
    st.sidebar.markdown("---")
    st.sidebar.info("Panel operativo para priorizacion de pruebas de software.")
    st.sidebar.caption("Lenguaje soportado: Java (.java)")

    st.header("Panel del Analista de Calidad")
    st.markdown(
        "Ingresa un repositorio de software en **Java** para extraer sus metricas en tiempo real "
        "y obtener el reporte de priorizacion de pruebas."
    )
    st.info("Alcance del modelo: Calibrado exclusivamente para proyectos desarrollados en **Java**, analizando archivos con extension `.java`.")

    tab_pred, tab_hist = st.tabs(["Solicitar Prediccion de Repositorio", "Historial de Reportes"])

    with tab_pred:
        opcion_entrada = st.radio(
            "Fuente del repositorio",
            [
                "Ingresar enlace remoto (GitHub / GitLab)",
                "Ingresar carpeta local (Repositorio privado en disco)",
                "Usar repositorio de muestra"
            ]
        )

        url_repo = ""
        df_repo = None

        if opcion_entrada == "Ingresar enlace remoto (GitHub / GitLab)":
            url_repo = st.text_input(
                "URL remota del repositorio",
                "https://github.com/apache/commons-cli"
            )
            st.caption("PyDriller clonara temporalmente e inspeccionara los commits del repositorio publico.")
        elif opcion_entrada == "Ingresar carpeta local (Repositorio privado en disco)":
            url_repo = st.text_input(
                "Ruta absoluta de la carpeta del repositorio en tu computadora",
                placeholder="Ejemplo: C:\\Users\\usuario\\proyectos\\mi-sistema-privado"
            )
            st.caption("PyDriller leera la carpeta oculta .git de tu disco duro de forma 100% offline y confidencial.")
        else:
            proyecto_muestra = st.selectbox("Selecciona un proyecto de prueba", [
                "apache/commons-cli (Procesador de comandos)",
                "apache/commons-lang (Utilidades de lenguaje)",
                "apache/commons-io (Operaciones de entrada y salida)"
            ])
            mapa_urls = {
                "apache/commons-cli (Procesador de comandos)": "https://github.com/apache/commons-cli",
                "apache/commons-lang (Utilidades de lenguaje)": "https://github.com/apache/commons-lang",
                "apache/commons-io (Operaciones de entrada y salida)": "https://github.com/apache/commons-io"
            }
            url_repo = mapa_urls[proyecto_muestra]

        limite = st.slider(
            "Cantidad maxima de commits a inspeccionar",
            min_value=50, max_value=500, value=150, step=50
        )

        if st.button("Ejecutar Analisis y Predecir Defectos"):
            if not url_repo or url_repo.strip() == "":
                st.warning("Por favor ingresa una URL remota valida o la ruta de una carpeta local.")
            elif opcion_entrada == "Ingresar carpeta local (Repositorio privado en disco)" and not os.path.exists(url_repo):
                st.error("La ruta especificada no existe en tu computadora. Verifica la ruta e intenta nuevamente.")
            else:
                with st.spinner("Inspeccionando repositorio y extrayendo metricas con PyDriller..."):
                    try:
                        df_repo = analizar_repositorio_en_vivo(url_repo, limite_commits=limite)
                    except Exception as e:
                        st.error(f"Error al analizar el repositorio: {e}")
                        df_repo = None

            if df_repo is not None and not df_repo.empty:
                st.success(f"Analisis completado. Se extrajeron {len(df_repo)} archivos Java.")

                valores_norm = escalador.transform(df_repo[['LOC', 'Complejidad', 'Churn']])
                X_pred = pd.DataFrame(valores_norm, columns=['LOC', 'Complejidad', 'Churn'])
                probabilidades = modelo_rf.predict_proba(X_pred)[:, 1]

                df_repo['Probabilidad_Defecto'] = probabilidades
                df_repo['Nivel_Riesgo'] = np.where(
                    df_repo['Probabilidad_Defecto'] >= 0.60, 'ALTO',
                    np.where(df_repo['Probabilidad_Defecto'] >= 0.40, 'MEDIO', 'BAJO')
                )

                df_repo['LOC_norm'] = valores_norm[:, 0]
                df_repo['CC_norm'] = valores_norm[:, 1]
                df_repo['Churn_norm'] = valores_norm[:, 2]
                df_repo = calcular_diagnostico_y_probabilidades(df_repo)

                df_repo = df_repo.sort_values(
                    by='Probabilidad_Defecto', ascending=False
                ).reset_index(drop=True)

                total_arch = len(df_repo)
                altos = (df_repo['Nivel_Riesgo'] == 'ALTO').sum()
                porc_alto = (altos / total_arch) * 100

                guardar_analisis_bd(url_repo, total_arch, altos, porc_alto)

                st.subheader("Resumen General de Riesgo")
                c1, c2, c3 = st.columns(3)
                c1.metric("Archivos Evaluados", total_arch)
                c2.metric("Archivos en Riesgo Alto", int(altos))
                c3.metric("Porcentaje Critico", f"{porc_alto:.1f}%")

                st.subheader("Reporte de Priorizacion de Pruebas")
                corte_prioridad = st.slider(
                    "Porcentaje de archivos con mayor riesgo a inspeccionar",
                    10, 100, 30, step=5
                )
                n_top = max(1, int(total_arch * (corte_prioridad / 100)))
                df_priorizado = df_repo.head(n_top)

                st.markdown(
                    f"**Top {corte_prioridad}% de archivos que deben revisarse primero "
                    f"({n_top} archivos):**"
                )

                st.dataframe(
                    df_priorizado[[
                        'Archivo', 'LOC', 'Complejidad', 'Churn',
                        'Probabilidad_Defecto', 'Nivel_Riesgo',
                        'Prob_Regresion', 'Prob_Logico', 'Prob_Rendimiento',
                        'Diagnostico_Riesgo'
                    ]].rename(columns={
                        'Prob_Regresion': 'Prob_Regresion (Churn)',
                        'Prob_Logico': 'Prob_Logico (CC)',
                        'Prob_Rendimiento': 'Prob_Rendimiento (LOC)'
                    }).style.format({
                        'LOC': '{:.0f}',
                        'Complejidad': '{:.0f}',
                        'Churn': '{:.0f}',
                        'Probabilidad_Defecto': '{:.1%}',
                        'Prob_Regresion (Churn)': '{:.1%}',
                        'Prob_Logico (CC)': '{:.1%}',
                        'Prob_Rendimiento (LOC)': '{:.1%}'
                    }),
                    use_container_width=True
                )

                defectos_en_corte = (df_priorizado['Nivel_Riesgo'] == 'ALTO').sum()
                efectividad = defectos_en_corte / altos if altos > 0 else 0.0

                st.info(
                    f"El corte del {corte_prioridad}% captura **{defectos_en_corte}** "
                    f"de los {int(altos)} archivos clasificados como ALTO riesgo. "
                    f"Efectividad del corte: **{efectividad:.1%}**"
                )

                guardar_reporte_prioridad(
                    url_repo, total_arch, corte_prioridad,
                    n_top, int(defectos_en_corte), efectividad
                )

                csv = df_priorizado.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Descargar Reporte en CSV", csv,
                    "reporte_priorizacion.csv", "text/csv"
                )
            else:
                st.warning("No se encontraron archivos Java modificados en el rango de commits indicado.")

    with tab_hist:
        st.subheader("Historial de Analisis Almacenados en SQLite")
        conn = sqlite3.connect("sistema_predictivo.db")
        df_hist = pd.read_sql_query(
            "SELECT id, fecha, repositorio, archivos_totales, "
            "archivos_alto_riesgo, porcentaje_riesgo "
            "FROM analisis_repositorios ORDER BY id DESC",
            conn
        )
        conn.close()
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
        else:
            st.info("No hay analisis guardados todavia en la base de datos.")
