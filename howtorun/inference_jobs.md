
You can run inference using the each finetuned talker by specifying the respective paths here.
reasoner model can be selected by passing the respective fireworks model path.
you need to update the file paths accordigly.
--talker-prompt
--reasoner-prompt
--talker-model-dir
--reasoner-model
--output-file


```bash
screen -S test_talker
screen -r test_talker
module load cuda/12.6
cd <project_path>
python talker_generate.py \
  --test-data ~/tr_data/test_mathdial.pkl \
  --talker-prompt talker_prompt.txt \
  --reasoner-prompt reasoner_prompt.txt \
  --talker-model-dir ~/finetuned_models/trt_qwen_2point5_7b/checkpoint-167 \
  # reasoner model path using fireworks
  --reasoner-model accounts/fireworks/models/deepseek-r1-basic \ 
  --output-file ~/tr_data/talker_outputs.jsonl --talker-model-short qwen_25_7b --reasoner-model-short deepseek_r1 2>&1 | tee ~/logs/test_talker_qwen_2point5_7b.log
```