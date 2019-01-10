[![Donate with Bitcoin](https://en.cryptobadges.io/badge/micro/1QJNhKM8tVv62XSUrST2vnaMXh5ADSyYP8)](https://en.cryptobadges.io/donate/1QJNhKM8tVv62XSUrST2vnaMXh5ADSyYP8)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://paypal.me/jarbasai)
<span class="badge-patreon"><a href="https://www.patreon.com/jarbasAI" title="Donate to this project using Patreon"><img src="https://img.shields.io/badge/patreon-donate-yellow.svg" alt="Patreon donate button" /></a></span>
[![Say Thanks!](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/JarbasAl)

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
