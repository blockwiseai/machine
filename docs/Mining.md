# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
2. [Registration ✍️](#registration)
3. [Setup ⚙️](#setup)
3. [Mining ⛏️](#mining)

## Before you proceed ⚠️

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml).

## Installation
> [!TIP]
> If you are using RunPod, you can use our [dedicated template](https://runpod.io/console/deploy?template=x2lktx2xex&ref=97t9kcqz) which comes pre-installed with all required dependencies! Even without RunPod the [Docker image](https://hub.docker.com/repository/docker/ericorpheus/zeus/) behind this template might still work for your usecase. If you are using this template/image, you can skip all steps below except for cloning.

Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/Zeus.git && cd Zeus
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). Note that after you run the last commands in the miniconda setup process, you'll be prompted to start a new shell session to complete the initialization. 

With miniconda installed, you can create a virtual environment with this command:

```bash
conda create -y -n zeus python=3.11
```

To activate your virtual environment, run `conda activate zeus`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command. This may take a few minutes to complete.

```bash
conda activate zeus
chmod +x setup.sh 
./setup.sh
```


## Registration

To mine on our subnet, you must have a registered hotkey.

*Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)*

To reduce the risk of deregistration due to technical issues or a poor performing model, we recommend the following:
1. Test your miner on testnet before you start mining on mainnet.
2. Before registering your hotkey on mainnet, make sure your port is open by running `curl your_ip:your_port`
3. If you've trained a custom model, test it's performance by deploying to testnet. Testnet performance is logged to a dedicated [Weights and Biases](https://wandb.ai/orpheus-ai/zeus-testnet).


#### Mainnet

```bash
btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 301 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Setup
Before launching your miner, make sure to create a file called `miner.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, **axon_port** and optionally **weather_xm_api_key**.


```bash
# Subtensor Network Configuration:
NETUID=18                                      # Network User ID options: 18,301
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                               # Endpoints:
                                               # - wss://entrypoint-finney.opentensor.ai:443
                                               # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings:
AXON_PORT=
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting
WEATHERXM_API_KEY=none                            # Only required if you participate in the local challenges
```

## Mining
Now you're ready to run your miner!

```bash
conda activate zeus
./start_miner.sh
```

### Challenge preference
Our subnet has two types of challenges: global gridded ERA5 prediction challenges OR local WeatherXM challenges. As a miner, you can only ever compete in **one** of these challenges, which is ensured by our validators. Effectively, we mimick having two separate metagraphs this way. To visualise this properly, please see our dedicated [MechaGraph Website](https://mechagraph.zeussubnet.com/) and switch the mechanism in the top right. Our unique logic prevents the two challenges from competing with each other, so each should be treated as a separate competition. If you as a team want to compete in both challenges, you therefore have to register multiple miners.

To specify your preference, miners are send a PreferenceSynapse every 5 minutes. By responding to this preference request, you can specify your desired challenge type. Not setting this will default to ERA5.

> [!TIP]
> See the dedicated ReadMe for the [incentive mechanism](IncentiveMechanism.md) for more information about the scoring on this subnet.

### Global ERA5 challenges
The datasource for this competition consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements across a multitude of variables. Miners will be sent a 2D grid of latitude and longitude locations and a time range **and** are asked to perform hourly forecasts at all those locations. Both the time interval and the geographical location will be randomly chosen, and both can be of **variable** size (within some [constraints](../zeus/validator/constants.py)). You will be asked to predict hourly measurements for these exact locations, for different numbers of hours. We focus on multiple variables, but each challenge will consist of just one variable to limit network data. For an up to date list of variables, see the [constants](../zeus/validator/constants.py) file.

The locations will be sent to you in the form of a **3D tensor** (converted to stacked lists of floats), with the following axes in order: `latitude`, `longitude`, `variables`. Variables consists of 2 values, with the first being the latitudinal coordinate and the second the longitudinal coordinate. Latitude is a float between -90 and 90, whereas longitude spans -180 to 180. Note that you will not be send a global earth representation, but rather a specific slice of this maximal range. 

Secondly, you will be send the start and end timestamp of the time-range you need to predict. These are formatted as float-timestamps, according to timezone GMT+0 (timezone used by Copernicus). These values will always be exactly rounded at the hour. You are also separately sent the number of hours you need to predict as an integer (you could calculate this yourself based on the timestamps). Note that both the start and end hour of the range are **included**.

Note that you do not need to send back the location data itself. The required return format is therefore a **3D tensor** (as a stacked list of floats) with the following axes: `requested_output_hours`, `latitude`, `longitude`. The value in each slot of this tensor corresponds to the prediction at that location. The [default miner code](../neurons/miner.py) illustrates exactly how to handle the input-output datastream of this competition. 

You will be scored based on the Root Mean Squared Error (RMSE) between your predictions and the actual measurement at those locations for the timepoints you were requested. The actual measurement is not yet known at the time that you receive the challenge, so you will be scored accordingly in the future when these data become available. Your goal is to minimise this RMSE, which will increase your final score and mechanism incentive. Depending on the difficulty of the challenge you are send (i.e. the variability of the region you need to predict), your RMSE is weighted differently to ensure fairer playing field.

The base miner already provides a decent level of predictions, but you will likely need to improve to stay competitive. It invokes [OpenMeteo's API](https://open-meteo.com/), which uses current environmental models to do forecasting. While improving upon this prediction might initially be challenging, just a small improvement will mean you can potentially obtain far higher rewards. Also note that you might run out of free-tier credits for this API eventually, depending on how often your miner is queried. 

### Hyperlocal WeatherXM challenges
The datasource here consists of the 9.7k stations spread around the globe by [WeatherXM](https://weatherxm.com/). These stations are located specifically at points of interest for customers, meaning a lot of value can be obtained by predicting properly here. For these challenges, the validator samples a random station, and sends the miner the latitude, longitude and elevation of this station. The miner is additionally send a time interval, and a **single** variable of interest (see the [constants](../zeus/validator/constants.py) file for all current variables) **and is asked to predict** that variable for the provided time interval (both bounds included). The timestamps are formatted as float-timestamps, according to timezone GMT+0. You will be asked to predict hourly measurements for the exact station location. 

Note that you do not need to send back the location data itself. The required return format is therefore a simple **list**, with a number for each `requested_output_hour`. The [default miner code](../neurons/miner.py) illustrates exactly how to handle the input-output datastream of this competition. 

The base miner already provides a decent level of predictions, but you will likely need to improve to stay competitive. It invokes WeatherXM's V1 API, which does require an API key. 







