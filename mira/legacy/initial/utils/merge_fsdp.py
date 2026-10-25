import sys
sys.path.append('.')  # Allow imports from the checkout root.

from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.model_merger.base_model_merger import ModelMergerConfig  # Checkpoint merger configuration.

def main():
    config = ModelMergerConfig(
        operation="merge",  # Alternatively, use the test operation.
        backend="fsdp",
        local_dir="checkpoints/verl_fsdp_gsm8k_examples/qwen2_5_0b5_fsdp_saveload/global_step_1/actor",  # Replace with the local FSDP checkpoint directory.
        target_dir="saved_models/qwen2.5-vl-3b_mira_data1_9k_210",  # Output directory for the merged model.
        # Optional Hugging Face upload setting.
        # Additional merger settings.
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

if __name__ == "__main__":
    main()
