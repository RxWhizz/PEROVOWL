# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the embedded local engine used by the Flutter app."""
import shutil
import sys
import tempfile
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).resolve().parent

# Config del generador/descubrimiento. El binario la busca como
# `config/generator.yaml`, pero lo que se empaqueta es la variante `.dist`, SIN
# las rutas de la maquina de desarrollo (Python de WSL, mounts de disco externo,
# directorios absolutos). Con `config/generator.yaml` a secas, una instalacion
# nueva heredaba `/home/luis/...` y `C:/NuevoVol/...` y fallaba de forma confusa
# en vez de decir "no configurado, corre el wizard de Entorno".
# PyInstaller conserva el nombre de origen, asi que se copia a un temporal ya
# renombrado.
_gen_dist = ROOT / "config" / "generator.dist.yaml"
if not _gen_dist.is_file():
    raise SystemExit(f"Falta {_gen_dist}")
_gen_stage = Path(tempfile.mkdtemp(prefix="perovowl-spec-")) / "generator.yaml"
shutil.copyfile(_gen_dist, _gen_stage)

datas = [
    (str(ROOT / "configs" / "monitor.example.yaml"), "configs"),
    (str(_gen_stage), "config"),
]

# Tabla de correccion scissor por elemento del sitio B. Sin ella,
# `buho.bandgap_scissor` no corrige el bandgap por acoplamiento espin-orbita y
# las etiquetas de entrenamiento salen crudas -- en silencio. Estaba fuera del
# bundle, asi que la correccion nunca se aplico en un binario publicado.
_soc = ROOT / "config" / "soc_scissor.json"
if _soc.is_file():
    datas.append((str(_soc), "config"))
else:
    raise SystemExit(
        f"Falta {_soc}. Generala con:\n"
        "  python scripts/calibrate_soc_scissor.py --out config/soc_scissor.json"
    )

# El pipeline en fuente. El runner de DFT no es codigo del binario: es un
# proceso aparte que lanza un Python externo (en Windows, el de WSL con GPAW), y
# ese interprete hace `sys.path.insert(ROOT/"src")` + `from buho import ...`.
# Nadie puede importar desde dentro del archivo de PyInstaller, asi que los
# fuentes tienen que existir como ficheros de verdad. Sin esto,
# `runner_launch_available` daba False en toda instalacion binaria y el DFT no
# se podia lanzar: la app quedaba en monitor + cribado.
#
# `_materializar_pipeline` en monitor_api.paths los copia a la raiz de datos al
# arrancar, que es donde el runner y WSL pueden leerlos.
_SCRIPTS_RUNTIME = [
    "buho_relax_runner.py",     # el runner de DFT
    "bench_machine.py",         # calibracion de slots/nucleos
    "buho_mlff_worker.py",      # worker del Tier 2
    "active_learning_orchestrator.py",
    "preconv_pbe_u.py",
]
for _nombre in _SCRIPTS_RUNTIME:
    _s = ROOT / "scripts" / _nombre
    if not _s.is_file():
        raise SystemExit(f"Falta el script de runtime {_s}")
    datas.append((str(_s), "pipeline/scripts"))

for _paquete in ("buho", "dft_cspbi3", "ml_surrogate"):
    _dir = ROOT / "src" / _paquete
    if not _dir.is_dir():
        raise SystemExit(f"Falta el paquete fuente {_dir}")
    for _py in _dir.rglob("*.py"):
        _rel = _py.relative_to(ROOT / "src").parent
        datas.append((str(_py), str(Path("pipeline/src") / _rel)))

estructuras = ROOT / "build" / "structures"
if estructuras.is_dir():
    datas.append((str(estructuras), "structures"))
else:
    raise SystemExit(
        "Faltan las estructuras preconvertidas. Ejecuta antes:\n"
        "  python scripts/pregenerate_structures.py"
    )

# `ase.spacegroup.crystal` lee `ase/spacegroup/spacegroup.dat` en tiempo de
# ejecución. PyInstaller recoge módulos Python, no los datos del paquete: sin
# esto, construir una estructura moría con FileNotFoundError. Solo los .dat —
# los otros 105 archivos de datos de ase son traducciones de su GUI.
datas += [(src, dst) for src, dst in
          collect_data_files("ase", includes=["**/*.dat"])
          if "test" not in Path(dst).parts]

for pkl in sorted((ROOT / "models").glob("surrogate_*.pkl")):
    datas.append((str(pkl), "models"))
for met in sorted((ROOT / "models").glob("surrogate_*.metrics.json")):
    datas.append((str(met), "models"))

hiddenimports = [
    # Los secretos se leen de .env. El import es perezoso, así que sin
    # declararlo aquí el binario podría quedarse sin él y no cargar
    # ninguna clave, en silencio.
    "dotenv",
    # `buho.structure.fases` lo importa dentro de `grupo_espacial()` para no
    # exigirlo a quien solo genera estructuras. PyInstaller analiza imports
    # estaticos, asi que no lo veia: v0.7.0 salio con el modulo de fases dentro
    # y sin spglib, de modo que identificar la fase --- que es lo que anunciaba
    # esa version--- devolvia "spglib no instalado" en todo binario.
    "spglib",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "itsdangerous",
    "sklearn.ensemble._forest",
    "sklearn.ensemble._gb",
    "sklearn.tree._tree",
    "sklearn.pipeline",
    "sklearn.preprocessing._data",
    "sklearn.impute._base",
    # Cadena de preparación de jobs DFT: todo se importa dentro de funciones,
    # así que el análisis estático de PyInstaller no lo alcanza.
    "ase",
    "ase.build",
    "ase.io",
    "ase.io.cif",
    "ase.spacegroup",
    "dft_cspbi3.structure_builder",
    "buho.structure.build_abx3",
    "buho.dft_jobs.prepare_relaxation_jobs",
]
if sys.platform != "win32":
    hiddenimports += ["uvicorn.loops.uvloop", "uvicorn.protocols.http.httptools_impl"]

excludes = [
    "matplotlib",
    "ase.gui",       # la GUI de ase no se usa y arrastra tkinter
    # `ase` NO se excluye: preparar los jobs DFT del cribado construye las
    # estructuras ABX3 en proceso (`buho.structure.build_abx3.build` importa
    # `ase.build` de forma diferida). Excluirla hacía que
    # POST /api/screening/runs/{id}/start-dft devolviera 500.
    "gpaw",
    "phonopy",
    # `spglib` NO se excluye: `buho.structure.fases.grupo_espacial` lo usa para
    # identificar en que fase quedo una estructura, que es lo unico que
    # distingue "medimos la fase" de "suponemos que es cubica". Estaba excluido
    # de cuando el binario no hacia cristalografia, y el exclude gana a
    # `hiddenimports`: tres versiones salieron sin el por no mirar esta lista.
    "tkinter",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "_pytest",
    "sphinx",
    "torch",
    "matgl",
    "dgl",
    # El motor es un servidor FastAPI sin ventana: en modo --engine el shell se
    # fuerza a "browser" y no_browser=True, así que Qt no se toca nunca.
    # PyInstaller lo arrastraba por los backends opcionales de pywebview —
    # 455 MB, el 53 % del bundle.
    "PyQt6",
    "PyQt5",
    "PySide6",
    "PySide2",
    "webview",
    # boto3/botocore entran como dependencia transitiva y nada del motor los usa.
    "boto3",
    "botocore",
    "s3transfer",
    "numpy.tests",
    "scipy.tests",
    "sklearn.tests",
    "pandas.tests",
    "pandas.plotting._matplotlib",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dft-monitor-engine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="dft-monitor-engine",
)
