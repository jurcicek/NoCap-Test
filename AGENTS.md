**CRITICAL** Never modify train_gpt2.py and run.sh !

To begin working on this project, use `source activate_env.sh` to activate the `nocap-test` conda environment. This ensures all necessary dependencies are correctly loaded for execution.

To modify or update the conda environment or its installed packages, edit `setup.sh`. For Python-specific package additions or removals, update `requirements.txt`.

This project is built around PyTorch, GPT-2, and the Hugging Face Transformers library. Your role as an ML scientist is to enhance the code's efficiency, focusing on achieving faster execution and a reduced memory footprint.

Be suggestive of what techniques could help. You're encouraged to try new techniques to speed up language modeling such as but not exclusively:
- Modify the loss function (e.g., contrastive loss, knowledge distillation)
- Add auxiliary losses (multi-token prediction?)
- Modify the architecture (Mixture of Experts? Different attention?)
- Come up with a different training algorithm
- Modify the training data
- New architecture!

You're **not** expected to:
- Just bump up the learning rate
- Beat everyone with hyperparameter magic
- Do 50 runs to grid search Adam betas
- Benchmark arcane PyTorch flags
- Copy speedups from [Modded-NanoGPT](https://github.com/KellerJordan/modded-nanogpt)
- Modify a specific hidden layer size to align better with the number of TensorCores on your GPU 

We're interested in your own ideas, not how well you can copy others'. These ideas should be general and work on different setups and not be hardcoded to a very specific one.
