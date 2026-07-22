#!/usr/bin/env bash

# Activate the project-local micromamba environment in the current Bash shell.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced: source scripts/activate_env.sh" >&2
  exit 2
fi

_fastwam_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_fastwam_project_root="$(cd -- "${_fastwam_script_dir}/.." && pwd)"
_fastwam_mamba_exe="${_fastwam_project_root}/.miniforge/_conda"
_fastwam_env_prefix="${_fastwam_project_root}/.conda/envs/fastwam-ood"

if [[ ! -x "${_fastwam_mamba_exe}" ]]; then
  echo "Missing project-local micromamba executable: ${_fastwam_mamba_exe}" >&2
  unset _fastwam_script_dir _fastwam_project_root _fastwam_mamba_exe _fastwam_env_prefix
  return 1
fi

if [[ ! -x "${_fastwam_env_prefix}/bin/python" ]]; then
  echo "Missing fastwam-ood environment: ${_fastwam_env_prefix}" >&2
  echo "Create the environment before activating it." >&2
  unset _fastwam_script_dir _fastwam_project_root _fastwam_mamba_exe _fastwam_env_prefix
  return 1
fi

export MAMBA_ROOT_PREFIX="${_fastwam_project_root}/.conda/mamba"
_fastwam_shell_hook="$("${_fastwam_mamba_exe}" shell hook --shell bash)" || {
  echo "Failed to initialize the project-local micromamba shell hook." >&2
  unset _fastwam_script_dir _fastwam_project_root _fastwam_mamba_exe _fastwam_env_prefix
  return 1
}
eval "${_fastwam_shell_hook}"
unset _fastwam_shell_hook

if ! micromamba activate "${_fastwam_env_prefix}"; then
  echo "Failed to activate: ${_fastwam_env_prefix}" >&2
  unset _fastwam_script_dir _fastwam_project_root _fastwam_mamba_exe _fastwam_env_prefix
  return 1
fi

unset _fastwam_script_dir _fastwam_project_root _fastwam_mamba_exe _fastwam_env_prefix
