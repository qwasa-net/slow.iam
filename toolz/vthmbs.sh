#!/bin/bash

AUTODIR=`mktemp -u -p ./data/vthmbs/`
OUTDIR=${3:-${AUTODIR}}
INDIR=${1:-.}
PATTERN=${2:-.+}
MPV=`which mpv`
MPV_FRAMES=${4:-12}
MPV_SSTEP=${5:-120}

mkdir -pv "${OUTDIR}"

echo "${INDIR} → ${OUTDIR} / ${PATTERN} / ${MPV} [${MPV_FRAMES},${MPV_SSTEP}]"

if [ ! -e "${INDIR}" ] || [ ! -d "${OUTDIR}" ] || [ ! -x "${MPV}" ];
then
    exit 1
fi

find "${INDIR}" -type f \
-regextype posix-egrep \
-iregex ".*${PATTERN}.*" \
-iregex ".*((wmv)|(avi)|(mpg)|(mpeg)|(mov)|(mp4)|(asx)|(flv)|(m4v)|(vob)|(mkv))$" \
-print | 
while read fn
do
    echo "${fn}"
    fnb=`basename "${fn}"`
    /usr/bin/mpv --really-quiet \
    --untimed --no-correct-pts --hr-seek=no --hr-seek-framedrop=yes --no-audio \
    -vo image --vo-image-format=jpg  --vo-image-jpeg-quality=95 \
    --vo-image-outdir="${OUTDIR}" \
    --start=1:00 --sstep="${MPV_SSTEP}" --frames="${MPV_FRAMES}" \
    "${fn}"

ls -1 "${OUTDIR}"/000000*.jpg | 
while read tn
do
    # echo "${tn}"
    tnb=`basename "${tn}"`
    mv "${tn}" "${OUTDIR}/${fnb:0:-4}-${tnb:4:10}"
done

done
