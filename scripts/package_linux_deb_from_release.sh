#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-0.3.0}"
PACKAGE_NAME="perovowl-dft-monitor"
DIST_TAR="$ROOT/dist/dft-monitor-desktop-$VERSION-linux-x86_64.tar.gz"
RELEASE_TAR="$ROOT/releases/v$VERSION/dft-monitor-desktop-$VERSION-linux-x86_64.tar.gz"
DEB_NAME="$PACKAGE_NAME-$VERSION-linux-amd64.deb"

if [ -f "$DIST_TAR" ]; then
  SOURCE_TAR="$DIST_TAR"
elif [ -f "$RELEASE_TAR" ]; then
  SOURCE_TAR="$RELEASE_TAR"
else
  echo "Missing desktop bundle: $DIST_TAR or $RELEASE_TAR" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STAGE="$TMP/stage"
EXTRACT="$TMP/extract"

mkdir -p \
  "$STAGE/DEBIAN" \
  "$STAGE/opt/$PACKAGE_NAME" \
  "$STAGE/usr/bin" \
  "$STAGE/usr/share/applications" \
  "$STAGE/usr/share/doc/$PACKAGE_NAME" \
  "$EXTRACT" \
  "$ROOT/dist"

tar -xzf "$SOURCE_TAR" -C "$EXTRACT"
cp -a "$EXTRACT/dft-monitor-desktop-$VERSION-linux-x86_64/." "$STAGE/opt/$PACKAGE_NAME/"

cat > "$STAGE/DEBIAN/control" <<EOF
Package: $PACKAGE_NAME
Version: $VERSION
Section: science
Priority: optional
Architecture: amd64
Maintainer: RxWhizz <noreply@github.com>
Depends: libc6 (>= 2.31), libgtk-3-0, libglib2.0-0, libstdc++6, libgcc-s1
Installed-Size: 416000
Homepage: https://github.com/RxWhizz/PEROVOWL
Description: PEROVOWL DFT Monitor desktop GUI
 Standalone desktop GUI for monitoring and steering the PEROVOWL/BUHO DFT workflow.
 Includes the frozen local monitor engine and surrogate model assets.
EOF

cat > "$STAGE/usr/bin/perovowl-dft-monitor" <<'EOF'
#!/bin/sh
APPDIR=/opt/perovowl-dft-monitor
export DFT_MONITOR_ENGINE="$APPDIR/engine/dft-monitor-engine"
export DFT_DATA_ROOT="${DFT_DATA_ROOT:-$HOME/PEROVOWL-data}"
cd "$APPDIR"
exec "$APPDIR/dft_monitor_flutter" "$@"
EOF

cat > "$STAGE/usr/share/applications/perovowl-dft-monitor.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=PEROVOWL DFT Monitor
GenericName=Monitor de simulaciones DFT
Comment=Desktop interface for the PEROVOWL/BUHO DFT workflow
Exec=perovowl-dft-monitor
Path=/opt/perovowl-dft-monitor
Terminal=false
Categories=Science;Physics;Chemistry;Monitor;
Keywords=DFT;GPAW;BUHO;PEROVOWL;perovskita;monitor;cribado;
StartupNotify=true
StartupWMClass=org.dft.monitor
EOF

cat > "$STAGE/usr/share/doc/$PACKAGE_NAME/copyright" <<EOF
PEROVOWL DFT Monitor package assembled from the v$VERSION release asset.
Source: https://github.com/RxWhizz/PEROVOWL
Bundled runtime components retain their upstream licenses.
EOF

chmod -R u+rwX,go+rX,go-w "$STAGE"
chmod 0755 \
  "$STAGE/DEBIAN" \
  "$STAGE/usr/bin/perovowl-dft-monitor" \
  "$STAGE/opt/$PACKAGE_NAME/dft_monitor_flutter" \
  "$STAGE/opt/$PACKAGE_NAME/engine/dft-monitor-engine"

dpkg-deb --build --root-owner-group "$STAGE" "$TMP/$DEB_NAME"
cp "$TMP/$DEB_NAME" "$ROOT/dist/$DEB_NAME"
# Desde dist/ y con el nombre pelado: si se le pasa la ruta absoluta, esa ruta
# acaba dentro de SHA256SUMS y `sha256sum -c` falla en la maquina del usuario,
# que no tiene /home/runner/... Igual que en build_desktop.sh y build_web.sh.
(cd "$ROOT/dist" && sha256sum "$DEB_NAME" > "$DEB_NAME.sha256")
dpkg-deb --info "$ROOT/dist/$DEB_NAME"
