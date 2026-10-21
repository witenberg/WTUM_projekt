#!/bin/sh
# Buduje APK, sprawdza zgodność z notebookami i instaluje na telefonie podpiętym po USB (jeśli jest).
set -e
cd "$(dirname "$0")"
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17}"
ADB="${ANDROID_HOME:-/opt/homebrew/share/android-commandlinetools}/platform-tools/adb"

./gradlew --console=plain -q testReleaseUnitTest assembleRelease
cp app/build/outputs/apk/release/app-release.apk ~/Downloads/LicznikPalcow.apk
echo "APK: ~/Downloads/LicznikPalcow.apk"

if "$ADB" get-state >/dev/null 2>&1; then
    "$ADB" install -r ~/Downloads/LicznikPalcow.apk
    "$ADB" shell am start -n pl.wtum.fingercounter/.MainActivity
else
    echo "Brak telefonu na USB. Przenieś plik ręcznie (Dysk Google, Quick Share, kabel)."
fi
