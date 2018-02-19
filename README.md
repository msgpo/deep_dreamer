wrapper for https://github.com/ProGamerGov/Protobuf-Dreamer

# usage

    model_path = "~/whatever_path"

    # check if model exists, if not download!
    DeepDreamer.maybe_download_and_extract(model_path)

    DD = DeepDreamer(model_path)

    saved_filepath = DD.dream(output_name, seed, channel_value,
                             layer_name, iter_value, step_size,
                             octave_value, octave_scale_value)

# install

    pip install TODO