# The start.sh banner: INKLING in block letters, with MIA AI LAB under it. Sourced by start.sh, which sets M, D and R.
banner() {
  local m=${M:-} d=${D:-} r=${R:-}
  printf '%s' "$m"
  cat <<'ART'
  ██╗███╗   ██╗██╗  ██╗██╗     ██╗███╗   ██╗ ██████╗
  ██║████╗  ██║██║ ██╔╝██║     ██║████╗  ██║██╔════╝
  ██║██╔██╗ ██║█████╔╝ ██║     ██║██╔██╗ ██║██║  ███╗
  ██║██║╚██╗██║██╔═██╗ ██║     ██║██║╚██╗██║██║   ██║
  ██║██║ ╚████║██║  ██╗███████╗██║██║ ╚████║╚██████╔╝
  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝
ART
  printf '%s  2 Mini Derisked · TensorFold · one DGX Spark%s\n' "$d" "$r"
  printf '%s  M I A   A I   L A B%s\n' "$m" "$r"
}
