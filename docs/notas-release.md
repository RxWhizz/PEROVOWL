# Monitor DFT 0.7.2

**El pipeline ya no supone que todo es cúbico.** Genera las fases que una
perovskita de haluro admite de verdad, deja que compitan en energía y **mide**
en cuál quedó el material en vez de decidirlo por decreto. Sobre CsPbI₃ la fase
cúbica pierde por 124 meV por fórmula, que es lo que dice el experimento.

Interfaz gráfica del pipeline de cribado de perovskitas: genera candidatos, los
criba con la cascada HTS, prepara y lanza los cálculos DFT, y sigue el progreso
en vivo.

## Descargar

Abre la pagina del release:

<https://github.com/RxWhizz/PEROVOWL/releases/tag/v0.7.2>

En **Assets**, descarga el paquete que corresponda a tu sistema:

| Sistema | Archivo recomendado | Uso |
|---|---|---|
| Windows 10/11 x64 | `dft-monitor-desktop-0.7.2-windows-x64.zip` | GUI de escritorio nativa con motor local embebido |
| Debian/Ubuntu x64 | `perovowl-dft-monitor-0.7.2-linux-amd64.deb` | GUI de escritorio instalable en el sistema |
| Linux x86_64 portable | `dft-monitor-desktop-0.7.2-linux-x86_64.tar.gz` | GUI portable sin instalador |
| Linux servidor/web | `dft-monitor-web-0.7.2-linux-x86_64.tar.gz` | Servidor local que abre la interfaz en navegador |

`SHA256SUMS` acompaña a los artefactos para verificar la descarga.

## Instalar y abrir

### Windows

Descarga `dft-monitor-desktop-0.7.2-windows-x64.zip`, descomprimelo **en una
carpeta corta** (p. ej. `C:\perovowl`) y ejecuta el `.exe` desde dentro de la
carpeta extraida:

```powershell
Expand-Archive .\dft-monitor-desktop-0.7.2-windows-x64.zip -DestinationPath C:\perovowl
C:\perovowl\dft-monitor-desktop-0.7.2-windows-x64\dft_monitor_flutter.exe
```

No necesita Python, Node, Flutter ni el repositorio. El motor local viaja dentro
de la carpeta `engine/`, que tiene que quedar **al lado** del `.exe`.

**Ruta corta a proposito**: el motor embebido anida directorios profundos y el
descompresor de Windows puede saltarse archivos por el limite de 260 caracteres
si extraes a `Descargas\...`.

**Si la app dice que no encuentra el motor**: casi siempre es el antivirus.
Windows Defender pone en cuarentena binarios de PyInstaller sin firmar como
`engine\dft-monitor-engine.exe`. Ve a *Seguridad de Windows -> Proteccion
antivirus y contra amenazas -> Historial de proteccion* y restaura/permite el
archivo. Alternativa: en la pestana **Diagnostico** de la app, "Seleccionar
motor" y apunta al ejecutable a mano.

### Debian/Ubuntu

Descarga `perovowl-dft-monitor-0.7.2-linux-amd64.deb` e instalalo con:

```bash
sudo apt install ./perovowl-dft-monitor-0.7.2-linux-amd64.deb
perovowl-dft-monitor
```

Tambien puedes abrirlo desde el menu de aplicaciones como **PEROVOWL DFT
Monitor**. Por defecto, el lanzador usa `~/PEROVOWL-data` como raiz de datos si
no defines `DFT_DATA_ROOT`.

### Linux portable

Si no quieres instalar el paquete `.deb`, usa el bundle portable:

```bash
tar xzf dft-monitor-desktop-0.7.2-linux-x86_64.tar.gz
./dft-monitor-desktop-0.7.2-linux-x86_64/dft_monitor_flutter
```

### Linux web/servidor

Para abrir la interfaz desde navegador o mirar el pipeline desde otra maquina:

```bash
tar xzf dft-monitor-web-0.7.2-linux-x86_64.tar.gz
./dft-monitor-web/dft-monitor-web --data-root /ruta/a/tus/datos
```

Se abre en `http://127.0.0.1:8000`. Con `--host 0.0.0.0` se expone en la red y
exige un token en `monitor.auth.token`.

## Verificar descargas

En Windows:

```powershell
Get-FileHash .\dft-monitor-desktop-0.7.2-windows-x64.zip -Algorithm SHA256
Get-FileHash .\perovowl-dft-monitor-0.7.2-linux-amd64.deb -Algorithm SHA256
```

En Linux:

```bash
sha256sum -c SHA256SUMS
```

## Qué trae

### Nuevo en 0.7.0

- **Fases candidatas en vez de una supuesta.** A partir de la perovskita cúbica
  se generan las distorsiones que esta familia admite —los sistemas de
  inclinación de Glazer— y se identifica el grupo espacial con spglib sobre la
  estructura ya relajada. La fase se **reporta**, no se asume: se reproducen
  Pm-3m, I4/mcm, P4/mbm, R-3c y Pnma desde la notación, sin tablas de Wyckoff.
- **Por qué importaba.** El pipeline construía siempre la cúbica. Para CsPbI₃
  esa es la fase α, que solo existe por encima de 330 °C; a temperatura ambiente
  el material está en otra, con otro bandgap. El propio código ya avisaba de
  ello sin poder demostrarlo.
- **El potencial elige la forma; la geometría calibrada, el tamaño.** Medido: el
  potencial ordena bien las fases, pero la celda que devuelve queda un 4.6 % por
  encima del experimento, y eso mueve el bandgap 0.75 eV. Se toma de cada uno lo
  que hace bien, y ambos parámetros de red quedan registrados para poder
  discutirlos.
- **La contracción del enlace B–X depende del haluro**, no solo del metal.
  Aplicar el factor de los yoduros a bromuros y cloruros comprimía la celda un
  2 % e **invertía el orden de los bandgaps** Cl > Br > I. Los cinco parámetros
  de red de referencia caen ahora dentro del 0.1 % del experimental.
- **El germanio se expande en lugar de contraerse**: su par solitario 4s empuja
  los haluros más lejos de lo que predice el radio iónico. CsGeBr₃ pasa de
  0.111 eV a 0.784.
- **GLLB-SC para el bandgap**, que calcula explícitamente la discontinuidad de
  la derivada —el término que a PBE le falta— en vez de parchearla. El error
  medio frente al experimento baja del 58.6 % al **28.7 %** sin ningún
  desplazamiento empírico.

### Nuevo en 0.6.0

- **El pipeline viaja dentro del binario.** El runner de DFT no es código de
  este proceso: es un Python externo —en Windows, el de WSL con GPAW— que
  importa `buho` desde ficheros. Nadie puede importar desde dentro del archivo
  de PyInstaller, así que hasta ahora `runner_launch` era `false` en **toda**
  instalación empaquetada y el DFT no se podía lanzar. Ahora los fuentes se
  copian a la raíz de datos al arrancar, y de paso llega también el script de
  calibración, que era inalcanzable desde un binario.
- **GPAW se instala desde la pestaña Entorno.** Crea el entorno en WSL con
  micromamba, fija numpy a 1.26 —GPAW 24.6 no compila contra la ABI de numpy 2,
  que es justo por lo que el entorno MLFF vive aparte—, comprueba los datasets
  PAW y verifica que todo importa.
- **Instalar WSL no se intenta.** Requiere administrador y reiniciar, así que la
  app lo detecta y da el comando exacto en vez de pedir elevación y fallar de
  forma confusa. Si ya tienes una distribución, se reutiliza en lugar de
  proponerte otra.
- Al terminar se escribe **dónde quedó el entorno** en tu configuración, con la
  raíz de datos traducida a la ruta que WSL ve (`/mnt/c/...`). Sin ese paso el
  entorno se creaba y nadie sabía encontrarlo.

### Nuevo en 0.5.0

- **Arranque rápido.** Un botón en **Inicio** encadena lo que antes había que
  hacer a mano y en orden: mide la máquina, fija el reparto de trabajos,
  comprueba que estén las piezas, enumera el espacio químico y arranca el bucle.
  Cada paso queda a la vista, y si falta algo se dice **antes** de empezar, con
  qué instalar — no a mitad de la primera ronda con el estado ya a medias.
- **Sondeo de la máquina en segundos.** Cuenta núcleos físicos (no hilos: GPAW
  en estas celdas está limitado por ancho de banda de memoria, y el SMT reparte
  el mismo ancho entre dos hilos), lee frecuencia y RAM disponible, y mide
  GFLOP/s reales — el recuento de núcleos miente en máquinas virtuales y en
  portátiles estrangulados por temperatura.
- **El reparto explica qué lo limitó.** «20 cálculos × 2 núcleos, limitado por
  la memoria» se puede discutir; «20 slots» a secas, no. La RAM manda: abrir más
  trabajos de los que caben hace paginar, y paginar cuesta más que un slot de
  menos.
- Es una **estimación**, y la interfaz lo dice: dimensiona a partir de recursos
  medidos. El óptimo real lo sigue dando el barrido de calibración, que mide
  t/iteración lanzando GPAW de verdad.

### De la serie 0.4

- **Protocolo de descubrimiento autónomo**: un ciclo ML → DFT → reentrenar →
  repetir que encadena rondas solo, sin volver a invocarlo entre medias.
  `buho active-learning discovery run` en consola, pestaña **Protocolo** en la
  app.
- **El aprendizaje activo ahora aprende**: cada reentrenamiento publica el
  modelo (`surrogate_bandgap_current.pkl`) y la siguiente ronda criba con él.
  Antes se reentrenaba y se descartaba — el bucle nunca usaba lo que acababa
  de aprender.
- **El bucle corre como proceso aparte**, no como hilo del servidor: cribar
  decenas de miles de candidatos con pandas ya no deja la API sin responder
  mientras tanto.
- **Sobrevive a un runner DFT que muere a media ronda**: lo detecta por falta
  de progreso (no solo por si el runner nunca llegó a arrancar), reintenta, y
  se rinde con un estado de error legible tras varios intentos en vez de
  colgarse indefinidamente.
- **Métrica honesta del reentrenamiento**: junto al `train_mae_eV` (que solo
  mide ajuste y baja artificialmente al crecer los datos) se registran
  `cv_mae_eV` (validación cruzada 5-fold) y `baseline_mae_eV` (predecir la
  media), para saber si el surrogate generaliza de verdad.
- **Pantalla Entorno**: una tarjeta por runtime (núcleo, API, GPAW, datasets
  PAW, MLFF) con lo que hay instalado, lo que falta y un botón para instalarlo,
  con el log en vivo. El equivalente en consola es `buho setup check` /
  `buho setup install`.
- **Tier 2 (MLFF/GNN) fuera del proceso del monitor**: `torch` + `matgl` +
  `pymatgen` pesan ~2 GB y en Windows son la parte más frágil de la pila. Ahora
  corren en su propio entorno —en Windows, dentro de WSL— y el monitor habla
  con él por un worker. `buho setup install mlff` lo crea.
- El entorno MLFF se crea **separado del de GPAW** a propósito: GPAW está fijado
  a numpy 1.26 y `matgl` exige numpy ≥ 2. Compartirlo rompería los cálculos.

### De antes

- **Vista en vivo** del pipeline: qué se está haciendo y tiempo estimado,
  calculado con la mediana real de los trabajos ya terminados.
- **Cribado HTS** en cascada de tres tiers, con malla de σ en el Tier 1.
- **Predictor** de bandgap con su incertidumbre y comparación contra DFT.
- **Candidatos** viables, verificados por DFT y ordenados por score
  fotovoltaico, exportables a CSV.
- **Visor de estructuras** 3D con bolas y palos, exportable a PNG.
- **Log de GPAW en vivo** mientras corren los cálculos.
- **Calibración de rendimiento**: mide cuántos trabajos concurrentes y cuántos
  núcleos por trabajo aguanta la máquina.

## Correcciones importantes

### Corregido en 0.7.2

- **spglib seguía sin viajar en el binario.** El arreglo de 0.7.1 lo declaraba
  en el spec de empaquetado, pero CI mantiene su propia lista de instalación a
  mano y no lo incluía: empaquetar no puede meter lo que no está instalado. La
  identificación de fase seguía sin funcionar. Ahora hay una prueba que compara
  ambas listas.

- **spglib no viajaba dentro del binario**, así que identificar la fase —lo que
  anunciaba 0.7.0— devolvía «spglib no instalado» en toda instalación
  empaquetada. `buho.structure.fases` lo importa dentro de la función para no
  exigirlo a quien solo genera estructuras, y PyInstaller analiza imports
  estáticos: no lo veía. Si instalaste 0.7.0, **actualiza**.

### Corregido en 0.7.0

- **El cribado comparaba magnitudes distintas.** El bandgap predicho está en la
  escala del cálculo y la ventana fotovoltaica en la del experimento. La
  consecuencia no era gradual sino total: de 3 793 candidatos, **los 3 793** se
  descartaban. El protocolo cribaba, no encontraba nada elegible y se declaraba
  terminado sin lanzar un solo cálculo.
- **El enumerador del protocolo se saltaba la cuantización de composiciones.**
  La corrección de 0.5.0 solo cubría uno de los dos caminos, y el que faltaba
  era justo el que gasta el DFT: 30 000 composiciones colapsaban en 1 998
  estructuras. El espacio real pasa de 86 035 candidatos a 3 793 estructuras
  distintas.
- **La validación cruzada repartía duplicados entre folds**, así que medía
  reconocimiento de repetidos y no generalización.
- Un modelo entrenado en otra escala **ya no se corrige como si fuera de la
  actual**: declara en cuál se entrenó, y si no coincide se avisa en vez de
  producir un número sin sentido con aspecto de estar bien.

### Corregido en 0.6.1

- **El reparto medido no surtía efecto hasta reiniciar.** El arranque rápido
  medía la máquina y escribía los slots en la configuración, pero esa se lee al
  abrir el proceso: el runner que ese mismo arranque lanzaba seguía usando los
  valores con los que se abrió la app. Medido sobre el binario 0.6.0 publicado —
  el fichero decía 19 trabajos en paralelo y el motor en marcha creía 2. Es lo
  contrario de lo que promete el botón, así que si usaste el arranque rápido en
  0.6.0, **actualiza**.

### Corregido en 0.6.0

- **Los datasets PAW se descargaban por costumbre.** Vienen ya en el paquete
  `gpaw-data` de conda-forge, en `site-packages`, no en `share/gpaw`. El
  instalador apuntaba al segundo, que no existe: habría dejado la ruta de
  setups señalando a un directorio vacío —y la app diciendo «no se encontraron
  setups PAW» tras una instalación correcta— además de bajar ~500 MB de más.
  Ahora comprueba y solo descarga si de verdad faltan.
- La documentación en Markdown sale del repositorio salvo los README y estas
  notas. No afecta al programa; reduce el ruido del árbol.

### Corregido en 0.5.1

Salido de auditar el binario publicado de 0.5.0 arrancándolo con un usuario
simulado, sin ninguna variable de entorno de desarrollo y desde `System32`.

- **El arranque rápido escribía el reparto donde nadie lo lee.** `runner_slots`
  y `runner_cores` iban a la raíz de `monitor.yaml`, pero el monitor recibe la
  subsección `monitor:`. El fichero quedaba válido, el botón decía
  «configuración ok» y los cálculos seguían corriendo con los slots de antes:
  justo lo que el botón promete ajustar. Es la corrección que motiva esta
  versión — si usaste el arranque rápido en 0.5.0, **actualiza**.
- **El binario llevaba dentro una ruta de la máquina de entrenamiento.**
  `models/surrogate_energy.metrics.json` guardaba el directorio absoluto de
  donde salieron los datos. No se leía nunca —es procedencia— pero viajaba a
  cada usuario. Ahora se guarda solo el nombre.

### Corregido en 0.5.0

Todo esto salió de mirar por qué el reentrenamiento decía que el surrogate era
veinte veces mejor que predecir la media.

- **Los dopantes diluidos desaparecían de la estructura.** Una supercelda 2×2×2
  tiene 8 sitios A, así que `round(0.051 × 8) = 0`: `Cs0.95Rb0.051SnI3` se
  construía como CsSnI3 puro, y su DFT se archivaba bajo la fórmula del dopado.
  Medido sobre el conjunto de entrenamiento: **56 de 111 filas compartían
  Eg = 1.075 eV repartidas en 48 fórmulas distintas**.
- **El generador proponía composiciones irrepresentables.** El muestreo continuo
  sacaba fracciones como 0.051 que ninguna supercelda de 8 sitios puede alojar.
  Ahora se ajustan a la rejilla de la celda, y la resolución sigue al tamaño de
  la supercelda si se agranda. De 40 000 composiciones que colapsaban en 2 204
  estructuras (**18×**) se pasa a **1 candidato = 1 estructura**.
- **Se gastaba DFT en repetir material.** El bucle descarta ahora los candidatos
  cuya estructura ya está calculada, y anota a cuál duplican.
- **El bandgap de DFT se usaba como característica para predecirse a sí mismo.**
  `Eg_target_eV` es `band_gap_gga_eV` más una constante por elemento B, y esa
  columna entraba como feature. En producción no existe —es justo lo que hay que
  predecir— y se rellenaba con 0.0 frente a un valor típico de 1.05.
- **La validación cruzada repartía duplicados entre folds**, así que medía
  reconocimiento de repetidos, no generalización. Con una fila por estructura el
  `cv_mae` real pasa de 0.0036 a **0.0330 eV** contra un baseline de 0.0363: el
  surrogate mejora un **9 %** sobre predecir la media, no un 92 %. Esa cifra
  honesta se muestra en el botón de arranque, porque es la que decide si vale la
  pena gastar días de DFT.
- **La celda se dimensionaba con la composición pedida**, no con la que
  realmente contenía: un dopante ausente seguía estirando la red.
- **Una especie que se queda sin átomos ya no pasa en silencio**: se avisa, y la
  metadata registra qué se perdió y qué composición se construyó de verdad.

### Corregido en 0.4.1

Todo esto salió de probar 0.4.0 en una Windows limpia y de auditar después el
patrón común: **un recurso que falta se resolvía a un valor neutro en vez de
quejarse**, así que el programa seguía dando resultados plausibles pero mal.

**Arranque en una maquina limpia**

- **La app no encontraba el motor.** El mensaje era de desarrollador («define
  DFT_MONITOR_ENGINE»). Ahora distingue si falta la carpeta `engine/` o si está
  pero falta el `.exe` —casi siempre el antivirus— y apunta al selector manual.
- **La raíz de datos caía en `C:\Windows\System32`.** Al abrir desde un acceso
  directo, el directorio de trabajo es ese, y ahí no se puede escribir ni hay
  configuración. Ahora usa `~/PEROVOWL-data`, igual que el paquete `.deb`.
- **El cribado fallaba con «No se encuentra .../config/generator.yaml».** No
  buscaba la copia que viaja dentro del binario.
- **El binario llevaba dentro rutas de la máquina de desarrollo** (`/home/...`,
  `C:/NuevoVol`, discos externos). Ahora se empaqueta una configuración limpia y
  los runtimes se configuran desde **Entorno**.

**Correcciones de física que no se estaban aplicando**

- **La corrección espín-órbita del bandgap no se aplicaba en ningún binario
  publicado.** La tabla de calibración ni viajaba en el paquete ni se buscaba
  donde estaba. El cribado etiquetaba con el bandgap de PBE crudo, en silencio.
- **El modelo reentrenado se escribía dentro de la instalación**, no en tus
  datos: se perdía al actualizar y fallaba si la app estaba en una carpeta de
  solo lectura. El ciclo de aprendizaje volvía a abrirse sin avisar.
- **Radio iónico del sitio A a la coordinación equivocada.** Se usaban valores de
  coordinación 6 donde toca coordinación 12. Familias enteras (Rb, K con Pb/Sn)
  quedaban fuera del espacio de búsqueda por un factor de tolerancia mal
  calculado.
- **La celda estaba un 9.7 % dilatada** respecto a la estructura real, lo que
  vale ~0.7 eV de error de bandgap — más que ignorar el espín-órbita.

**Que ahora se dice en vez de callarse**

- Los candidatos puntuados con **valores por defecto** (porque faltó una
  predicción) quedan marcados; antes eran indistinguibles de los medidos.
- Si un modelo no carga —causa habitual: versión de scikit-learn— se dice cuál y
  por qué, en vez de devolver predicciones vacías.
- Un cálculo DFT cuyo resumen no se pudo leer ya **no cuenta como convergido**.
- El **riesgo de politipo** se marca: el factor de tolerancia dice si los iones
  encajan en una perovskita cúbica, no si esa es la fase estable a temperatura
  ambiente. Confirmarlo exige fonones.

**Instalación desde fuentes**

- `pip install -e .` traia scikit-learn 1.9, con el que los modelos incluidos no
  se pueden cargar. El tope estaba solo en CI; ahora protege tambien al usuario.

### Nuevo en 0.4.0

- **Una ronda ya no muere por una dependencia opcional**: si falta el entorno
  MLFF, el cribado sigue con los tiers 0 y 1 y lo dice en pantalla, en vez de
  reventar a mitad. Se descarta menos material, pero el DFT —que es lo caro— no
  se bloquea.
- **El estado dejaba de poder rearrancarse**: al fallar durante el cribado, el
  estado persistido se quedaba en «cribando» para siempre. Parecía que seguía
  trabajando cuando el hilo ya había muerto, y el botón de ejecutar salía
  deshabilitado. Ahora cualquier fallo deja el estado diagnosticable.
- **Barra de navegación**: con once secciones ya no cabían en la ventana por
  defecto y las últimas quedaban inalcanzables. Ahora la barra se desplaza.

### De antes

- **Datasets PAW**: la ruta estaba escrita a mano en siete archivos y dejó de
  existir; todos los cálculos fallaban al arrancar. Ahora se resuelve
  verificando que el dataset esté de verdad, y se aborta con instrucciones si
  falta, en vez de consumir el lote entero trabajo a trabajo.
- **Geometría ABX3**: la red cúbica se fijaba con `2√2·(r_B+r_X)`, que es la
  relación A–X; la correcta es `2·(r_B+r_X)`, que fija el enlace B–X. Las
  estructuras salían √2 veces dilatadas.
- **Runners y app**: cerrar la ventana mataba los cálculos en marcha. Los
  procesos largos ya no heredan los descriptores del motor.
- **Seguimiento de lotes**: el monitor se quedaba mirando el lote configurado al
  arrancar y no veía los lanzados después.

## Limitación conocida

El protocolo autónomo puede declararse `done` tras pocas rondas aunque queden
miles de candidatos sin verificar: la ventana fotovoltaica del Tier 1 asume
bandgap experimental, y el surrogate aprende el bandgap PBE de la criba, que es
sistemáticamente más bajo. No es un fallo del código — es una calibración
pendiente. Detalle y opciones de arreglo en
[#7](https://github.com/RxWhizz/PEROVOWL/issues/7).

## Licencias

Incluye ASE (LGPL-2.1-or-later) dentro del binario. El resto de dependencias
empaquetadas —scikit-learn, numpy, scipy, pandas, FastAPI, uvicorn, psutil— es
BSD/MIT.
