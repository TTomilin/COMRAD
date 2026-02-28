#!/bin/bash
set -euo pipefail
cleanup() {
    echo -e "TERM"
    pkill -P $$ || : # Kill the child
    exit 1
}
trap cleanup EXIT

wads=("scenarios"/*.wad)
for i in "${!wads[@]}"; do
    printf "%d %s\n" "$((i+1))" "${wads[$i]}"
done

while true; do
    if ! read -r -p "Select a scenario (1-${#wads[@]}): " choice; then
        echo ""
        exit 0
    fi

    if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#wads[@]} )); then
        wad="${wads[$((choice-1))]}"
        break
    fi
done

./vizdoom -windowed 1 -iwad freedoom2.wad -file "$wad" -host 2 &
hpid=$!
sleep 0.5
./vizdoom -windowed 1 -iwad freedoom2.wad -file "$wad" -join 127.0.0.1 &
jpid=$!
echo $hpid
echo $jpid

wait $hpid
wait $jpid
