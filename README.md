# PPO for MxNet
*This code also origionally worked on with Chien-chein Huang and Sean Welleck as
part of a larger project*

`Please note, this code was ripped out of a larger project and as such the organization
may not make perfect sense. I will clean this up if I have time`

The goal of this repository is to create running [PPO](https://openai.com/blog/openai-baselines-ppo/)
code (completely based off
of OpenAI's [baselines](https://github.com/openai/baselines)) using MxNet, as
no suitable implementation seems to be around.

I have added a main function to train a simple cart-pole example.

The following commands should be sufficient to set up an environment and run
the code (skip packages you may have already installed).

```
> python3 -m venv mxenv
> source mxenv/bin/activate
(mxenv) > pip install --upgrade pip
(mxenv) > pip install mxnet matplotlib gym
(mxenv) > python ppo_main.py
```
