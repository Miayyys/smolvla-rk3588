"""Small public command dispatcher; implementations live in functional packages."""
import argparse
import runpy
import sys
from qvla.paths import source_path

# Each entry reuses the existing implementation and forwards its original options.
COMMANDS = {
    'prepare_data': {'download': ('download_and_upload.py', []), 'split': ('split_libero.py', []), 'simulator': ('prepare_libero_sim.py', [])},
    'benchmark_hardware': {'tables': ('build_hardware_tables.py', []), 'compile': ('benchmark_cost_compile.py', []), 'board': ('benchmark_cost_board.py', []), 'formats': ('probe_rknn_numeric_formats.py', [])},
    'search': {'run': ('run_haq_local_loop.py', []), 'space': ('build_haq_search_space.py', []), 'freeze': ('freeze_haq_candidates.py', []), 'compare': ('compare_haq_rl_random.py', [])},
    'distill': {'inputs': ('prepare_openvla_teacher_inputs.py', []), 'teacher': ('cache_openvla_teacher.py', []), 'review': ('review_teacher_labels.py', []), 'train': ('qat_train_haq.py', ['--mode', 'fp-distill'])},
    'quantize': {'qat': ('run_first8_v2_qat.py', ['--config', 'config/qat_distilled_v1_no_teacher_v1.json']), 'qat-train': ('qat_train_haq.py', ['--mode', 'qat']), 'ptq': ('pack_v2_ptq_control.py', [])},
    'convert': {'vision': ('convert_smolvla_vision_rknn.py', []), 'export': ('export_selected_qat_rknn.py', []), 'rknn': ('compile_selected_qat_rknn.py', []), 'partition': ('run_v1_partition_conversion.py', []), 'language-export': ('export_smolvla_rkllm_language.py', []), 'language': ('compile_rkllm_full_prefill_probe.py', [])},
    'evaluate': {'gpu': ('eval_distill_qat_libero.py', []), 'board': ('run_smolvla_board_libero.py', []), 'resources': ('benchmark_smolvla_board_resources.py', []), 'report': ('report_final_model_comparison.py', []), 'replay': ('verify_v1_partitioned_replay.py', [])},
}

def main(group):
    entries=COMMANDS[group]
    parser=argparse.ArgumentParser(description=f'QVLA {group}: select a stage; remaining options go to its implementation.')
    parser.add_argument('stage', choices=entries)
    # Parse only the stage, so "stage --help" shows the implementation's options.
    if len(sys.argv)==1 or sys.argv[1] in ('-h','--help'):
        parser.print_help();return
    selected=parser.parse_args(sys.argv[1:2]).stage
    filename,defaults=entries[selected]
    path=source_path(filename)
    sys.argv=[str(path),*defaults,*sys.argv[2:]]
    runpy.run_path(str(path),run_name='__main__')
