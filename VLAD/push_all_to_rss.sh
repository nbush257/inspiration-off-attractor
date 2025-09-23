
# Parse --dry-run flag
DRY_RUN=""
for arg in "$@"; do
	if [[ "$arg" == "--dry-run" ]]; then
		DRY_RUN="--dry-run"
	fi
done

SRC_DIRS=(hering_breuer low_d model physiology singlecell)
DEST_BASE="/data/rss/helens/ramirez_j/ramirezlab/nbush/projects/VLAD/VLAD"
INCLUDE_OPTS=(--include='*/' --include='*.pdf' --include='*.csv' --include='*.pqt' --include='*.mp4' --exclude='*')

for dir in "${SRC_DIRS[@]}"; do
	rsync -rzP $DRY_RUN "${INCLUDE_OPTS[@]}" ./$dir/ "$DEST_BASE/$dir"
done
