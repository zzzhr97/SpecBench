# Configuration

This directory contains three configuration files used to set up models and TTD methods.

1.  `config_model.yaml`

    This file specifies the model configuration for the **generation stage**. As described in the [README](https://github.com/zzzhr97/SpecBench/blob/main/README.md), the `model` parameter serves as a key for retrieving the corresponding configuration from this file. Example format:
    
    ```yaml
    Qwen3-14B-thinking:
      model_name: Qwen/Qwen3-14B
      max_tokens: 4200
      temperature: 0.7
      top_p: 0.8
      extra_body:
        top_k: 20
        min_p: 0.0
        chat_template_kwargs:
          enable_thinking: true
    ```

    Explanation of the format:

    - The first line (`Qwen3-14B-thinking`) is the unique name specified by the `model` parameter. This name is also used to determine the path where model results are stored. Avoid using `/` in the name.
    - `model_name` is the actual model identifier passed to the API server.
    - All other parameters are unpacked and forwarded to the API’s `chat.completions.create` method, defining the decoding behavior.

    With this setup, we can save results of the same model under different decoding configurations by simply changing the `model` parameter.

2. `config_eval_model.yaml`

    This file follows the same structure as `config_model.yaml`, but it defines the decoding parameters for the **evaluation stage** evaluator model (`eval_model`). Typically, parameters are set to 0 during evaluation.

3. `config_ttd.yml`

    This file contains the hyperparameter configurations for different TTD methods. For details on specific parameters required by each method, refer to `specbench/ttd.py`.
