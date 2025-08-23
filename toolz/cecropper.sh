#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 {basedir} {mode}"
    exit 1
fi

BASEDIR=`echo "$1" | sed 's#^file://##i'`
CMD=$2
FORCE=$3

find "${BASEDIR}" -maxdepth 1 -type f |
while read fn
do

    iw=$(identify -format "%w" "${fn}")
    ih=$(identify -format "%h" "${fn}")

    if [ -z "${iw}" ] || [ -z "${ih}" ]; then
        echo "skipping ${fn} => no dimensions"
        continue
    fi

    if [ $iw -eq $ih ]; then
        echo "skipping ${fn} == ${iw}x${ih}"
        if [ "x${FORCE}" == "x" ]; then
            continue
        fi
    fi

    csize=$(( iw>ih?ih:iw ))

    if [ "${CMD}" == "noop" ]; then  # no operation
        echo "${fn}: ${iw}x${ih}"
        continue
    elif [ "${CMD}" == "cq" ]; then  # central square crop
        offx=$(( (iw - csize) / 2 ))
        offy=$(( (ih - csize) / 2 ))
    elif [ "${CMD}" == "lq" ]; then  # left square crop
        offx=0
        offy=0
    elif [ "${CMD}" == "rq" ]; then  # left square crop
        offx=$(( iw - csize ))
        offy=$(( ih - csize ))
    elif [ "${CMD}" == "gcc" ]; then  # golden central center crop
        csize=$(( 30 * csize / 45 ))
        offx=$(( (iw - csize) / 2 ))
        offy=$(( (ih - csize) / 2 ))
    elif [ "${CMD}" == "gbc" ]; then  # golden bottom center crop
        csize=$(( 30 * csize / 45 ))
        offx=$(( (iw - csize) / 2 ))
        offy=$(( ih - csize ))
    elif [ "${CMD}" == "gtc" ]; then  # golden top center crop
        csize=$(( 30 * csize / 45 ))
        offx=$(( (iw - csize) / 2 ))
        offy=0
    else
        echo "Invalid mode: $2"
        exit 1
    fi

    echo "${fn}: ${iw}x${ih} → ${csize}x${csize} (+${offx}+${offy})"

    mv "${fn}" "${fn}.bak"
    convert "${fn}.bak" -crop "${csize}x${csize}+${offx}+${offy}" "$fn" && rm "${fn}.bak"
    identify "${fn}"

done