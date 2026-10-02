# Abstract

An architectural drawing describes how a space is meant to be used. The people who use it move in ways the drawing never records, and once a space is occupied that behaviour is usually lost. Motion Pixels asks whether it can be kept. The thesis investigates whether pedestrian trajectories can be predicted from the relationship between human behaviour and architectural space, and whether the result can become a layer of evidence an architect can read alongside the plan.

The method turns ordinary video into measured movement. Pedestrians are detected and tracked, the footage is calibrated to the architectural plan by homography so that movement is expressed in real metres, and each trajectory is encoded in terms of both its motion and its spatial situation, its distance to obstacles and boundaries. A recurrent model, a Long Short-Term Memory network chosen for its stability over long rollouts, then predicts movement forward from a short window of observed motion. The work was developed first on a single sandbox site, the esplanade in front of MACBA in Barcelona, and then on a dataset of five calibrated recordings across the city containing 3,534 tracked trajectories.

An early failure shaped the research. The first models flattened the curved movement the sandbox was chosen to capture. A controlled capacity test showed that the architecture could represent angular movement once it had enough examples, which identified the problem as primarily a shortage of data rather than a flawed design. Prediction was then evaluated across several horizons, from roughly one to twenty metres of travel.

The results are bounded and consistent. Short-range prediction is reliable, with an average displacement error near half a metre at the shortest horizon and endpoint placement inside a one metre tolerance about eighty-three percent of the time. Accuracy falls as the horizon grows, and predictions tend to straighten and fall short of the real path. Later experiments that changed the training objective recovered a large part of the missing shape, which confirmed that the limits were as much about how the model learned as about how much data it had. The models were evaluated on unseen people within the same sites, not on entirely unseen spaces, and no claim of transfer beyond the studied sites is made.

The contribution is less a model than a way of working. Prediction is used as a test of whether movement carries usable spatial structure, and it does, most clearly at the scale where an architect reasons about a threshold, an edge, or a crossing. Observed and predicted movement, together with behavioural maps of flow, speed and density, form an additional layer of architectural information that returns lived behaviour to the design process rather than leaving it in memory.

**Keywords:** pedestrian movement, trajectory prediction, behavioural mapping, architectural analysis, computer vision

---

*Target: 500 words maximum. Actual: see chapter audit in 01_SOURCE_MAP.md.*
