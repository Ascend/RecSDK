#!/bin/bash
# Package script
# Copyright © Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
set -e

CURDIR=$(dirname "$(readlink -f "$0")")
SCRIPT_NAME=$(basename "$0")
ROOT_PATH=$(readlink -f "$CURDIR"/../)
OUTPUT_PATH="$ROOT_PATH/output"
VERSION_FILE=${SCRIPT_DIR}/../../mindxsdk/build/conf/config.yaml
get_version() {
  if [ -f "$VERSION_FILE" ]; then
    VERSION=$(sed '/.*mindxsdk:/!d;s/.*: //' "$VERSION_FILE")
    if [[ "$VERSION" == *.[b/B]* ]] && [[ "$VERSION" != *.[RC/rc]* ]]; then
      VERSION=${VERSION%.*}
    fi
  else
    VERSION="5.0.T104"
  fi
}

get_version
export VERSION


function make_zip_package()
{
    cd "${OUTPUT_PATH}"
    pkg_file=$(ls "$OUTPUT_PATH"/*"${1}"*."${2}")
    pkg_file="${pkg_file##*/}"
    pkg_release="${pkg_file%."${2}"}"

    package_file="${OUTPUT_PATH}"/package
    [ -d "$package_file" ] && rm -rf "$package_file"
    mkdir "$package_file"
    cp -f "${OUTPUT_PATH}"/crldata.crl "$OUTPUT_PATH/${pkg_release}.${2}.crl"
    cp "$pkg_release".* "$package_file"

    cd "$package_file"
    chmod 600 "$pkg_release.${2}"
    chmod 600 "$pkg_release.${2}".cms
    chmod 600 "$pkg_release.${2}".crl
    zip_file="${3}$pkg_release.zip"
    zip -r "$zip_file" "$pkg_release.${2}" "$pkg_release.${2}".cms "$pkg_release.${2}".crl

    mv "$package_file/$zip_file" "${OUTPUT_PATH}/$zip_file"
    echo "zip $zip_file success !"
    [ -d "$package_file" ] && rm -rf "$package_file"
    return 0
}

function main()
{
    make_zip_package Ascend-mindxsdk-mxrec tar.gz
    return 0
}

echo "begin to execute $SCRIPT_NAME"
main;ret="$?"
echo "finish exuecte $SCRIPT_NAME, result is $ret"
