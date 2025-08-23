#!/bin/bash

INDIR=${1:-.}
INDIR=`echo "${INDIR}" | sed 's#^file://##i'`
PATTERN=${2:-.+}

AUTODIR=$( mktemp --dry-run --tmpdir=./data/vthmbs/ )
OUTDIR=${3:-${AUTODIR}}

MPV=`which mpv`
MPV_FRAMES=${4:-12}
MPV_SSTEP=${5:-120}

HWDEC=${6:-auto-safe}

mkdir -pv "${OUTDIR}"

TMPDIR=$( mktemp --directory )
echo "dirs: out=${OUTDIR}; tmp=${TMPDIR}"

echo "${INDIR} → ${OUTDIR} / ${PATTERN} / ${MPV} [${MPV_FRAMES},${MPV_SSTEP}]"

if [ ! -e "${INDIR}" ] || [ ! -d "${OUTDIR}" ] || [ ! -x "${MPV}" ];
then
    exit 1
fi

find "${INDIR}" -type f \
-regextype posix-egrep \
-iregex ".*${PATTERN}.*" \
-iregex ".*((wmv)|(avi)|(mpg)|(mpeg)|(mov)|(mp4)|(asx)|(flv)|(m4v)|(vob)|(mkv)|(ts))$" \
-print |
while read fn
do
    echo "${fn}"
    fnb=`basename "${fn}"`

    /usr/bin/mpv --really-quiet \
    --untimed --no-correct-pts --hr-seek=no --hr-seek-framedrop=yes --no-audio --slang= \
    --hwdec=${HWDEC} -vo image --vo-image-format=jpg --vo-image-jpeg-quality=98 \
    --vo-image-outdir="${TMPDIR}" \
    --start=0:35 --sstep="${MPV_SSTEP}" --frames="${MPV_FRAMES}" \
    "${fn}" 2>&1 | strings

    ls -1 "${TMPDIR}"/*.jpg | wc -l
    rename "s/0000/${fnb:0:-4}-/" "${TMPDIR}"/0000*.jpg
    mv "${TMPDIR}"/* "${OUTDIR}"

done

rm -rfv "${TMPDIR}"
