#!/bin/sh
# Add a player to the arena: upload.sh <arena url> <name> <spec> [--replace] <file>...
# The spec is <model>:<file> for an ONNX export (send the .onnx and its .json)
# or rpsi:<command> for an engine (send a .tar.gz holding it). From outside
# the LAN set ARENA_ADMIN_TOKEN.
set -e
url=$1; name=$2; spec=$3; shift 3
replace=0
[ "$1" = "--replace" ] && { replace=1; shift; }
set -- -F "name=$name" -F "spec=$spec" -F "replace=$replace" "$@"
args=""
for f in "$@"; do
  case "$f" in -F|name=*|spec=*|replace=*) args="$args $f" ;; *) args="$args -F files=@$f" ;; esac
done
[ -n "$ARENA_ADMIN_TOKEN" ] && args="-H X-Admin-Token:$ARENA_ADMIN_TOKEN $args"
curl -sS -f $args "$url/api/players"
echo
