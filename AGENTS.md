# Instructions

When analyzing the code, assume that we are using flash attention. This has tremendous effect on the speed
and memory consumption.

# Definition

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


# Documentation

## Documentation Structure and Organization

When creating new documentation for features, testing, profiling, or any project-related content, follow these guidelines:

### Directory Structure
- **Location**: Always create documentation in the `docs/` directory
- **Naming Convention**: Create new subdirectories with numbered prefixes (e.g., `01-feature-name/`, `02-another-feature/`)
- **Sequential Numbering**: Use incremental numbers following the existing subdirectory numbering pattern
- **Descriptive Names**: Use clear, descriptive names after the number prefix

### Documentation Types
Create comprehensive documentation for:
- **Features**: New functionality, optimizations, or architectural changes
- **Testing**: Test suites, validation procedures, and quality assurance
- **Profiling**: Performance analysis, bottleneck identification, and optimization reports
- **Experiments**: Research findings, experimental results, and analysis
- **Tutorials**: Step-by-step guides for using or extending the project
- **Architecture**: System design, component relationships, and technical specifications

### Content Standards
- **Completeness**: Document all relevant aspects of the feature or analysis
- **Clarity**: Use clear, concise language with proper grammar and spelling
- **Structure**: Organize content with appropriate headings, bullet points, and code examples
- **Examples**: Include practical examples, code snippets, and usage demonstrations
- **Cross-references**: Link related documentation and provide clear navigation paths

### File Organization within Subdirectories
- Use descriptive filenames (e.g., `README.md`, `IMPLEMENTATION.md`, `RESULTS.md`)
- Include a main `README.md` file in each subdirectory explaining the purpose and contents
- Organize related files logically (e.g., separate files for code, results, and analysis)
- Maintain consistent file naming conventions across all documentation

### Version Control
- Keep documentation up-to-date with code modifications
