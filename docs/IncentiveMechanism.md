# Incentive Mechanism

This notebook aims to highlight how incentive is distributed in this subnet. In version 2.0.0, we have implemented two unique types of challenges, which are represented on the blockchain as **mechanisms**. Crucially, validators enforce that each miner can only compete in one type of challenge, meaning we effectively have two 'metagraphs' (called **Mechagraphs**) / competitions. 

This document will first outline our metagraph structure, after which the incentive for each separate competition will be outlined.

## Metagraphs and Mechagraphs
Due to the nature of Bittensor, our subnet still has a single metagraph where all neurons live and where registrations happen. However, unlike any other subnet currently, we support 512 neurons. For each of the two competitions, we reserve 128 slots for miners. Validators keep track of score for everyone who responds, but only set weights for the top 128 miners per competition. Additionally, validators will only ever set weight for one competition (mechanism) per miner, to ensure miners have to choose which one to compete in.

The validator neurons themself are part of the remaining 256 slots, so they are not counted towards the 128 for either competition. The roughly ~245 slots that remain are called the **safe zone**. These neurons will not get any incentive (until they beat someone in the top-128 per competition), but will also not immediately be deregistered. When you register a miner, you will therefore always kick out someone in this zone, and not in the actual competition. If a miner falls out of the top 128 slots, there will be quite some time before it goes completely to the bottom, since the **safe zone** is substantially large. Therefore this incentive mechanism elimates registration pressure in favour of a oligarchic 'winner'-takes-all, with 128 winners per competition. The reason for this decision over having for example a single winner (or a few winners) is that our API's and products keep the flexibility of querying a bunch of addresses in parallel which are all known to perform well (part of the top 128).

### Visualisation
> [!TIP]
> Checkout the custom [Mechagraph dashboard](https://mechagraph.zeussubnet.com/)

We fully appreciate that our interpretation of mechagraphs does not visualise cleanly in the default metagraph shown on for example TaoStats. We have therefore developed a custom [Mechagraph dashboard](https://mechagraph.zeussubnet.com/), which includes proper Mechagraphs for each competition. By default, it will show the full metagraph in a similar fashion to TaoStats. However, through the selection box in the top right, it is possible to select the separate mechagraph for each mechanism. 

If a mechagraph is selected, only the top 128 miners (based on current mechanism incentive) are shown, without the **safe zone** and the validators for example. This will allow miners to more cleanly compare their performance against relevant competitors, as opposed to miners in the other mechanism. The emission split between both mechanisms is also visualised. Note that the website currently points to Testnet, but will be updated accordingly after the Mainnet update.

## Challenges and baselines
This section will provide a very high level overview of the challenges themselves, some parts of which are similar for both ERA5 and WeatherXM challenges. First, we will outline what kind of challenges are send for each competition. Additionally this includes information about the baseline you will be compared against.

> [!NOTE]
> For a **detailed breakdown** of the input and output data formats, please see the [Mining Guide](Mining.md).

### ERA5
The datasource for this competition consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). 
- **Input send to miner**: You are send a rectangular bounding box of locations, a start and end time, and a single variable to predict.
- **Target output**: You are asked to predict hourly measurements for each of those locations, for that specific variable. Both start and end time should be included.
- **Baseline**: Your score is compared against OpenMeteo's strong API with default parameters, so the goal is to outperform this API. 

### WeatherXM local challenges
The datasource here consists of the 9.7k stations spread around the globe by [WeatherXM](https://weatherxm.com/).
- **Input send to miner**: You are send the latitude, longitude and elevation of a single station. You are also send a start and end time, and a single variable to predict.
- **Target output**: You are asked to predict hourly measurements for that station for that specific variable. Both start and end time should be included.
- **Baseline**: Your score is compared against WeatherXM V1 forecast API with default parameters, so the goal is to outperform this API. 

## Scoring 
Having introduces the types of challenges and the baselines, this section will briefly elaborate how you are scored. The underlying principles are actually similar for both competitions, so they are grouped here. Scoring for a particular challenge is based on *Root Mean Squared Error* (RMSE). This error is calculated between your prediction, and the actual ground truth. To acquire this ground truth, the validator has to wait until it becomes available. Once the RMSE for your prediction is calculated, we also calculate the RMSE for the baseline outlined for your challenge in the previous section. 

Your raw score is then based on your percentage of improvement (i.e. lower RMSE) compared to this baseline. To account for differences in difficulty between challenges at different times and locations, we actually use a modified version of the RMSE throughout, known as a **difficulty-weighted RMSE**. 

Additionally, we reserve a small amount of the score for the miner's efficiency, so their response speed. Scores are also normalised relative to other miners for that particular challenge. 

Once your score for a particular challenge has been calculated, your overal score is based on a exponential moving average (EMA). We actually use a custom implementation with a dynamic $\alpha$, based on how long the miner has been in the competition. Please see [this notebook](ScoreAdjustment.ipynb) for more explanation on the moving average aggregation of the score.

> [!NOTE]
> For a **detailed mathematical breakdown** of the scoring system, please see the [dedicated notebook](DifficultyScoring.ipynb).

