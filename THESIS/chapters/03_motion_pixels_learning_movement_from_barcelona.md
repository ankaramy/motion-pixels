# Chapter 3. Motion Pixels, learning movement from Barcelona

The previous chapter described the pipeline in principle. This one describes what happened when it was pointed at real footage of real streets. The work did not arrive fully formed. It began with a single site used as a testing ground, ran into a clear failure, and only then grew into a dataset spread across Barcelona. I have kept that order here, because the mistakes are part of the argument. They are what turned a rough idea into a method.

## MACBA, the sandbox experiment

### The sandbox

The first site was the esplanade in front of the Museu d'Art Contemporani de Barcelona, MACBA. Anyone who has been there knows it less as a museum forecourt than as one of the best known skate spots in Europe. The wide, smooth plaza fills through the afternoon with skateboarders, and around them a normal crowd of pedestrians crosses, gathers and watches.

[FIGURE 3.1 HERE]
Source file: THESIS/figures/booklet/MACBA_2011.jpg | THESIS/figures/booklet/source_bcncolours_macba011.jpg
Proposed caption: The MACBA esplanade in the Raval. The plaza was never designed as a skate park, but its smooth ground and low ledges made it one, the negotiation between design and use this thesis is about. Photographs courtesy of MACBA, Museu d'Art Contemporani de Barcelona.

I chose it on purpose. A pipeline meant to read movement needs movement worth reading, and skateboarding produces exactly the range that ordinary walking does not. Fast runs and slow rolls, sharp turns, sudden stops, curved lines that double back on themselves. If the tools could handle the variety of speeds and angles on that plaza, they could probably handle a calmer square. So one video shot from the museum toward the esplanade became the sandbox, the ground on which every part of the pipeline was tested for the first time.

MACBA sits in the Raval, and its plaza was never designed as a skate park. It became one. That fact is the thesis in miniature, a designed space repurposed by the behaviour it happened to afford, its smooth ground and low ledges reading as an invitation the architects never wrote. Filming there meant starting the research on exactly the kind of gap between intention and use that motivated it in the first place.

Calling it a sandbox is deliberate. Nothing about this stage was meant to prove that the method generalised. It was meant to prove that the method ran at all, end to end, from raw footage to a trajectory placed on the plan and a prediction rolled forward from it. Everything that follows in this chapter is a response to what that first site showed.

[FIGURE 3.2 HERE]
Source file: THESIS/figures/booklet/sandbox_experiment/skate_1_tracking_still_frame_01002.png | THESIS/figures/booklet/sandbox_experiment/skate_2_tracking_still_frame_00911.png
Proposed caption: Tracked pedestrians and skateboarders on the MACBA esplanade, the sandbox recording. The mix of fast rolls, sharp turns and ordinary walking gave the range of speeds and angles the pipeline was built to handle.

One step in that workflow matters here. As each frame is processed, detected faces are blurred before anything is stored, so the sandbox, and every recording after it, worked only with anonymous bodies and their movement, never with identifiable people.

### Behavioral metrics

Before anything could be predicted, the movement had to be described. Each test run over the sandbox produced a set of behavioural measures, and these are the quantities that make a trajectory legible as behaviour rather than as a line.

The simplest are about presence and pace. How many people are active at once, usually one or two clearly tracked individuals in a given window. How many are paused, standing still for a moment. How many are dwelling, staying in one area long enough to count as occupying it rather than passing through. Dwelling and lingering matter to an architect because they mark the places a space invites people to stop, which is often where a design succeeds or fails.

Then there is direction. A trajectory is not only where a person went but how their heading changed along the way. Counting direction shifts gives a sense of how much a path bends, which separates a straight crossing from a wandering one. Lingering zones, the areas where dwelling concentrates, come out of the same data seen from above.

Speed is described in three ways. The average speed of a person across their path, their maximum speed, and the number of stops they make. From these the movement is sorted into slow, medium and fast, so that a plaza can be read not just as busy or quiet but as a mix of paces. In the behavioural maps these categories are given fixed thresholds, with the boundaries between slow, medium and fast set at 0.8 and 1.6 metres per second.

None of these measures is exotic. They are close to what an architect already notices when they stand in a space and watch it, who is rushing, who is loitering, where the crowd knots. What changes is that here they are computed for every tracked person at once and tied to a position on the plan, so the noticing becomes a record that can be compared between sites and across times of day. These metrics are not the prediction. They are the description the prediction is later tested against, and they are already, on their own, a way of seeing a space that a plan does not offer.

[FIGURE 3.3 HERE]
Source file: THESIS/figures/booklet/sandbox_experiment/skate_1_bottleneck_heatmap_original.png | THESIS/figures/booklet/sandbox_experiment/skate_1_flow_field_quiver_original.png | THESIS/figures/booklet/sandbox_experiment/skate_1_linger_zones_plot_original.png
Proposed caption: Early behaviour maps from the MACBA sandbox, a bottleneck heatmap, a flow field and linger zones. These first rough versions confirmed that the tracking already carried readable spatial structure, before the method moved out to the other sites.

### Prediction models

The sandbox was also where the choice of model was made. Trajectory prediction can be approached with several architectures, and rather than assume one, I compared four families drawn from the literature reviewed in the previous chapter. A Long Short-Term Memory network (LSTM), a Gated Recurrent Unit (GRU), a Temporal Convolutional Network (TCN), and a small Transformer.

Each was trained on the sandbox trajectories under the same conditions, and their behaviour was compared not only on training loss but on how they held up when rolled forward step by step. That distinction turned out to matter more than the headline loss. A model that predicts a single next step well can still drift badly once its own predictions are fed back in to produce a longer path. The comparison used a common setup, each model trained for thirty epochs with a batch size of 1024, optimised with Adam against a mean squared error on the next displacement. Holding the training identical meant the differences that showed up were differences between the architectures, not differences in how hard each had been pushed.

The LSTM gave the most stable trajectory prediction, with the lowest endpoint drift and the strongest control over error as it accumulated across a rollout. The GRU reached a slightly lower training loss but showed more drift once it was predicting autoregressively. The TCN reached the lowest loss overall and yet performed worst on prediction stability, which is the quality that actually matters when a path is rolled out over many steps. The small Transformer did not offer enough of an advantage on this scale of data to justify its added complexity.

So the project settled on the LSTM, and the reason shaped everything after. The goal here is not the lowest error on a single step. It is a predicted path that stays coherent as it extends, because that is what an architect would actually look at. The LSTM was chosen for its behaviour over distance, not for a number on a validation set.

There is a broader lesson in this that shaped the rest of the project. The best model in the abstract is not always the best model for a small, specific dataset. Transformers dominate the public benchmarks in part because those benchmarks are large. On a few thousand trajectories from one plaza, a simpler recurrent model that degrades gracefully was the more honest choice, and it stayed the choice as the dataset grew.

[FIGURE 3.4 HERE]
Source file: mp-data/outputs/prediction/experiments/phase-2b-final/phase2b_final_spatial_model_comparison.png
Proposed caption: The model families compared on the sandbox. The LSTM holds a rolled out path together with the least endpoint drift, which is why it was chosen over the lower loss but less stable GRU and TCN.

### Overfit test and feature ablation

The sandbox did not only confirm what worked. It exposed the problem that shaped the rest of the research.

When the trained model was asked to predict, its outputs collapsed toward straight lines. Faced with the turning, curving movement that made MACBA interesting in the first place, the model smoothed it away and predicted roughly forward motion. A skater carving a long arc came back as a short straight stub. Watching the model erase precisely the movement I had chosen the site to capture was the low point of the early work, and also the moment the research found its real question. Was the architecture incapable of representing angular movement, or was it simply not seeing enough of it to learn from?

To separate those two possibilities I ran a controlled capacity test. The original sandbox held a small number of tracked trajectories, 52 in total, drawn from roughly eighteen thousand recorded positions. That is very little data. So the dataset was duplicated tenfold, exposing the same motion patterns to the model many times, not to prove generalisation but to ask a narrower question. Given enough exposure to these exact patterns, can the architecture reproduce their angular structure at all?

It could. On the duplicated data the best configuration recovered the turning behaviour that had been lost. Its average displacement error fell to about 0.05 metres, and, more tellingly, its curvature correlation with the real paths rose to around 0.50 and its cumulative heading matched the ground truth almost exactly. On the un-duplicated held-out data the same schema had shown a curvature correlation near zero, about negative 0.03, which is essentially no relationship. The contrast between those two numbers is the whole point. The architecture was capable of representing angular movement. It had simply lacked the data to learn it.

The capacity result is a memorisation test. A model that has seen the same paths ten times and then reproduces them has proven that the schema can fit that kind of motion, not that it will predict new motion in a new place. It would be wrong to read the low capacity error as evidence of real world accuracy. What it licenses is a narrower and still useful claim. The early failure was primarily a data problem, not a flaw in the model design, and the way forward was to grow the dataset rather than abandon the architecture.

The same experiment was used to choose the feature schema. Four schemas were compared, from a minimal one carrying only motion to a full one carrying motion, normalised position and relational spatial encodings. The four carried six, eight, ten and eleven features. On the held-out set their differences were small and did not favour the richer schemas, the minimal motion-only model reaching an average displacement error of about 0.49 metres and the ten-feature schema about 0.53. On its own that looks like a case against spatial features. Set beside the capacity result, it says something different, that the held-out set was simply too small to reward the extra information. The configuration named Model C, with ten features combining ego motion, position within the site, and distance to obstacles and boundaries, gave the strongest capacity behaviour while staying tied to the architectural question, and it was kept for both reasons.

[FIGURE 3.5 HERE]
Source file: mp-visualization/overfit10x_replots/overfit10x_modelC_highlight_collage.png
Proposed caption: The capacity test. With the sandbox data duplicated tenfold, Model C, highlighted, reproduces the curved paths that the model had flattened before. Axes are normalised, and this is memorisation evidence that the schema can fit angular motion, not a claim of predictive accuracy.

[FIGURE 3.6 HERE]
Source file: THESIS/figures/booklet/feature_ablation/traj_00_id24004.png | THESIS/figures/booklet/feature_ablation/traj_02_id12007.png | THESIS/figures/booklet/feature_ablation/traj_03_id28007.png
Proposed caption: Feature ablation on the sandbox. Each panel overlays the schemas, from motion only up to the full Model C set, on the same ground truth path. The richer schemas recover the turning that the minimal one flattens, which is the capacity result that justified keeping the spatial features.

### Dataset structure agreed on

What came out of the sandbox was a settled description of each moving person, and it is the schema every later experiment uses. Each step of a trajectory carries ten values. Two describe the immediate motion as a displacement in each direction. One is the speed. Two encode the heading as its sine and cosine, so that direction is continuous rather than jumping at north. One is the turn rate. Two give the normalised position within the calibrated site. The last two are the distances to the nearest obstacle and to the boundary of the walkable area.

The choice to encode heading as a sine and cosine pair rather than a single angle is a small technical point with a real effect. An angle jumps from 359 degrees back to zero, and a model reading that jump sees a discontinuity where the movement was smooth. Splitting it into two continuous values removes the seam. Details like this are where a schema either respects the geometry of movement or fights it.

Read as a whole, the schema says something simple. A person's next movement is treated as a function of how they were just moving, where they are in the space, and what lies immediately around them. That is the hypothesis from the first chapter turned into columns of a dataset.

[FIGURE 3.7 HERE]
Source file: THESIS/figures/booklet/figure_3_7_model_c_feature_schema.png
Proposed caption: The agreed Model C feature schema, ten values per step. Ego motion, position in the site, and proximity to obstacles and boundaries, the three registers the model reasons over.

<!-- TRANSITION -->

The sandbox settled two things. It showed that the pipeline runs end to end, and it showed that the early failure was a shortage of data rather than a limit of the model. It also fixed the feature schema that every later experiment uses. What it could not do was prove that any of this generalises, because it was one site seen many times over. The obvious next move was to widen the ground under the method, from a single recording to many, so the data could carry the variety that one plaza never could.

<!-- /TRANSITION -->

## Building the dataset

Seeing that the sandbox had worked, one thing became evident, and that was that it needed more data. I had to find a way to initiate this process. I moved from one site to many, and the second half of this chapter is about that larger dataset and what it produced.

I went site hunting across Barcelona, and the selection was not random. The end goal was variety, because a model that is trained on one kind of space learns only of that space and little else. And in that variety I found spatial diversity. MACBA and the esplanade of Espanya gave open plaza movement. The Montjuïc stairs gave vertical, constrained movement and the angular paths that stairs force on people. Open squares such as Plaça Montjuïc, Plaça Espanya and Plaça Catalunya gave large, loosely structured crossings. The Red Bridge on Passeig de Colom gave something the others did not, a curved structure that bends movement around it.

The reason to spread the dataset across such different spaces is not only technical. It is the architectural claim that is being tested. If movement were purely a matter of individual intention, the space would not matter and one plaza would train the model as well as five. The bet behind Motion Pixels is the opposite, that the space affects movement, and the only way to see that mark is to hold many spaces side by side and ask whether the model reads them differently. A single site could never answer that.

Each site stresses the method differently. A staircase constrains where people can go, so the spatial features carry a lot of the signal. An open plaza does the opposite, giving people freedom and making the recent motion the stronger cue. A curved structure tests whether the model can follow a bend rather than cut across it. Some of these differences are visible before any model runs. On the esplanade, movement spreads out and slows, and people trace long, loose diagonals across open ground. On the Montjuïc stairs, the architecture does most of the deciding, and the trajectories compress into the few lines the steps allow. Plaça Catalunya, the busiest of the five, layers many crossing flows over one another, which makes it rich and also noisy. The Red Bridge bends its traffic along a curve, precisely the kind of shaped movement the model finds hardest to reproduce.

[FIGURE 3.8 HERE]
Source file: THESIS/figures/booklet/dataset/sites_mass_plan/esplanade-espanya.png | THESIS/figures/booklet/dataset/sites_mass_plan/placa-catalunya.png | THESIS/figures/booklet/dataset/sites_mass_plan/red-bridge-combined.png | THESIS/figures/booklet/dataset/sites_mass_plan/stairs-montjuic-1.png | THESIS/figures/booklet/dataset/sites_mass_plan/placa-montjuic.png
Proposed caption: The site plans of the recordings, drawn to the same convention. From open plazas to a staircase and a curved bridge, the set was chosen so the difficulty is spread across the same spatial features the schema encodes.

The site hunting was framed around roughly eight recordings, from which on the order of seven thousand trajectories were extracted, with a smaller number expected to survive culling into training. The dataset that the final experiments were actually trained and evaluated on is more specific. It contains five recordings and 3,534 tracked trajectories, after two recordings were excluded for quality reasons.

Getting from raw footage to these trajectories was the largest single effort in the project. Each recording had to be run through detection and tracking, checked, recalibrated when the mapping drifted, and encoded against its mask. The seven thousand figure from the acquisition stage and the 3,534 that survived into training describe the two ends of that funnel, raw extraction on one side and quality controlled, calibrated trajectories on the other.

The five recordings that make up the final dataset, known here by their working names, are the esplanade of Espanya, Plaça Catalunya, Plaça Espanya, the Montjuïc stairs, and the Red Bridge. Their sizes are uneven, and that reflects how busy each space was, the frequency of its usage and how long it was filmed.

| Recording | Tracked positions | Trajectories |
|---|---:|---:|
| esplanade_espanya_01 | 283,120 | 522 |
| placa_catalunya_01 | 355,583 | 1,255 |
| placa_espanya_01 | 212,224 | 824 |
| stairs_montjuic_01 | 127,605 | 334 |
| red_bridge_combined_01 | 85,847 | 599 |

[FIGURE 3.9 HERE]
Source file: THESIS/figures/booklet/dataset/esplanadeespanya_tracking.png | THESIS/figures/booklet/dataset/placacatalunya_tracking.png | THESIS/figures/booklet/dataset/placaespanya_tracking.png | THESIS/figures/booklet/dataset/stairsmontjuic1_tracking.png | THESIS/figures/booklet/dataset/redbridge_tracking.png
Proposed caption: A frozen tracking frame from each of the five recordings, with live trajectories and per-track metrics overlaid. The same pipeline meets a different kind of movement at each site, from the loose diagonals of the esplanade to the compressed lines of the stairs.

Plaça Catalunya alone contributes more than a third of the trajectories, and the Montjuïc stairs the fewest, it is a small space with way less exposition than its counterparts. A model trained on this mix sees far more open plaza movement than stair movement, and that shows up later in where the predictions are strongest. I did not rebalance the data, partly because the imbalance reflects how busy these spaces actually are, and partly because forcing an even split would have meant discarding real movement to flatter the model.

Two further recordings, a second Montjuïc stairs clip and a Plaça Montjuïc clip, were prepared but left out of training because their tracking or calibration did not meet the standard the others held and therefore would hinder the training. A recording was kept if its tracking held identities cleanly enough and its calibration placed movement on the plan without obvious drift. The two that were dropped failed one of those tests badly enough that including them would have added noise dressed up as data. In a larger project this culling would itself be worth studying, because the boundary between usable and unusable footage is where a lot of the real difficulty of this method lives.

Preparing these sites was not automatic. Each recording needed its own calibration, matching points in the footage to points on the plan, and its own spatial mask marking the walkable area, the obstacles and the boundaries. The masks were drawn by hand for each site, which is slow, but it is what lets the spatial features mean something specific to each space rather than being generic.

[FIGURE 3.10 HERE]
Source file: mp-data/annotations/manual_masks_v3/placa_catalunya_01/overlay_manual.png
Proposed caption: A hand drawn walkable and obstacle mask over Plaça Catalunya. Masks like this were made for every recording, and they let the model describe a pedestrian by their real relationship to the architecture around them.

One more decision matters for how the results are read, and it concerns how the data was split. The final baseline model was evaluated with a track-held-out split, where whole trajectories are divided into training, validation and test sets, 2,827 for training and roughly 350 each for validation and test, and every recording appears in all three. This tests whether the model generalises to unseen people within the same set of spaces. It does not test whether it generalises to an entirely unseen site, which is a harder question and a different experiment, and the difference between the two matters when reading the results later.

The baseline trained on this dataset, the model the horizon experiments build on, is a two layer recurrent network with a hidden size of 128, reading a ten step window of motion and predicting the next displacement. It was trained with the AdamW optimiser and a scaled error on that next step, over several hundred thousand training windows drawn from the trajectories. The architecture is deliberately modest. The interesting variation in this thesis is in the data and the training objective, not in ever larger models.

None of this makes the dataset large by machine learning standards. Three and a half thousand trajectories is small. The value is not in its size but in what each trajectory carries, a real path in a real, calibrated space, described in terms an architect can read.

<!-- TRANSITION -->

With five calibrated recordings in place, the project had something it did not have before, a body of real movement described in the same terms across very different spaces. Before asking a model to predict from it, that description is worth looking at directly. The same encoded features that feed the prediction also draw a picture of how each space is used, and those pictures hold value on their own. The next section turns the dataset into behavioural maps, the readable images that sit between a raw table of trajectories and a forecast.

<!-- /TRANSITION -->

## The behavioral maps

Before the prediction results, the dataset produces images that are useful on their own. The behavioural maps turn thousands of trajectories into a single readable picture of how a space is used. Each one takes a different feature from the schema and gives it back to the eye.

A plan shows a space as it was designed. A behavioural map shows the same space as it was used, drawn from the movement of everyone the camera saw. Laying one over the other is the idea at its simplest. The maps are not predictions and they are not models. They are description, and description is already more than architecture usually keeps once a space is occupied.

The flow field renders the direction of movement across a site. Trajectories are drawn and coloured so that the dominant lines of travel become visible, the routes people actually take rather than the ones the plan suggests. It answers a plan reading question directly. Where does this space channel movement, and where does it leave it diffuse. The animated version of this map is a stylised, density directed representation. It is designed to communicate the sense of flow, not to replay each person's measured heading frame by frame, and I describe it that way to avoid implying a precision the animation does not carry.

What the flow field cannot show is why a line forms where it does. That reading is left to the architect, who can see whether a dominant route follows an entrance, avoids an obstacle, or cuts a corner the design did not intend. The map narrows the question. It does not answer it.

[FIGURE 3.11 HERE]
Source file: mp-visualization/behavior_maps/behaviormaps_final/flow_fields/placa_catalunya_01_flow_fields_still.png
Proposed caption: The flow field for Plaça Catalunya. The strongest lines of travel emerge from the accumulated trajectories, showing where the space concentrates movement and where it disperses it.

The speed map, built from the speed metric, shows one dot per pedestrian coloured by how fast they were moving, using the slow, medium and fast categories described earlier. Read across a whole site, it separates the parts of a space where people hurry from the parts where they slow down and settle. The two often correspond to something in the architecture, an edge, a threshold, a place with a reason to pause. The thresholds that sort the dots are fixed rather than relative, so a slow dot means the same pace on every site. That makes the maps comparable. A crowd that reads as mostly fast on the esplanade and mostly slow in a tighter square is telling you something about how the two spaces are used, not about how the colour scale was set. The Montjuïc stairs read as a field of slower dots, movement checked by the steps, while the open esplanade carries far more fast crossings.

[FIGURE 3.12 HERE]
Source file: THESIS/figures/booklet/dataset/placacatalunya_speedmap.png | THESIS/figures/booklet/dataset/placaespanya_speedmap.png | THESIS/figures/booklet/dataset/esplanadeespanya_speedmap.png
Proposed caption: Speed maps across three of the sites, one point per pedestrian coloured by pace. Plaça Catalunya, the busiest and the most pertinent to highlight, shows the clearest separation between the fast lines people use to cross and the slow pockets where they gather.

The density map reads congestion. It scores areas by how heavily they are used and warms their colour accordingly, so that the pinch points and crowded cells stand out. This is the occupancy reading the project produces. The scores are precomputed from the trajectory data, and the map describes where use concentrated rather than diagnosing a given point as a circulation failure. Congestion is the behaviour most directly tied to how a space performs, and the one a plan is worst at predicting, since two corridors of equal width can behave completely differently once real flows meet in them.

[FIGURE 3.13 HERE]
Source file: THESIS/figures/booklet/dataset/placacatalunya_heatmap.png | THESIS/figures/booklet/dataset/placaespanya_heatmap.png | THESIS/figures/booklet/dataset/stairsmontjuic1_heatmap.png
Proposed caption: Bottleneck density across three sites. Warmer cells mark where movement concentrated. In the open plazas the pressure spreads along the main crossing lines; on the Montjuïc stairs it compresses into the narrow band the steps allow.

Each map is really a view onto one column of the dataset. The flow field reads direction, the speed map reads the speed feature, the density map reads position and dwell. This is deliberate. The same encoding that feeds the prediction model also feeds the images, so the maps and the forecasts are two readings of one description rather than two separate products. A feature that turned out to carry little signal for prediction could still make a legible map, and the reverse is also true.

One caveat runs across all of the maps. They are built from tracked movement, and tracking is imperfect. A dropped identity or a jittery path leaves a faint trace in the aggregate. At the scale of thousands of trajectories these errors mostly wash out. At the scale of a single line they do not, so the maps are best read as strong description rather than exact measurement.

The last family, the prediction projections, shows the model's output rather than the observed data. A predicted path is drawn over the plan from a seed of real motion, so that the forecast can be read in the same architectural frame as everything else. At their best, over an open plaza, these projections lay a plausible near future onto the space, and they are the clearest expression of what the whole pipeline is for. A hero version, a long horizon prediction set over two adjoining plazas, was made to carry the idea in a single image. Taken together, the maps are the point where the dataset stops being a table and becomes something an architect can look at and argue with.

[FIGURE 3.14 HERE]
Source file: mp-visualization/hero_visuals/h400_dual_placas/hero_H400_placa_catalunya_placa_espanya_background.png
Proposed caption: A prediction projection set over two adjoining plazas. Observed history and predicted continuation are drawn in the same architectural frame as the plan, which is the reading the whole pipeline is built to produce.

[FIGURE 3.15 HERE]
Source file: THESIS/figures/booklet/dataset/esplanadeespanya_predictionrollout.png | THESIS/figures/booklet/dataset/placacatalunya_predictionrollout.png | THESIS/figures/booklet/dataset/placaespanya_predictionrollout.png
Proposed caption: Prediction projections over three site plans. Each draws a long, illustrative rollout of the model forward from observed motion, laid onto the plan so the forecast can be read in the same frame as the design. At this length they should be read as tendency, not as an exact route.

<!-- TRANSITION -->

The maps describe what already happened. They show where a space channels movement, where it slows people down, where it crowds. That description is the first half of the thesis in practice, and for many architectural questions it is enough on its own. The harder claim is the predictive one. If the movement carries real structure, a model should be able to take a short slice of it and continue it, and that continuation should hold against what the person actually did. The next section tests exactly that, across a range of distances, and it is where the method meets its limits.

<!-- /TRANSITION -->

## Predictive Layer: Horizon Rollouts

Once the dataset was stable, the real question was how far ahead the model could see. And that's exactly what a horizon is: It is how many steps into the future the model is asked to predict from a short slice of observed motion. I trained and evaluated five of them. In the notation I use through this chapter they are H20, H60, H100, H200 and H400, which correspond to roughly 1, 3, 5, 10 and 20 metres of travel. The numbers are frames, not minutes. A person at a normal pace would cover that ground in a few seconds, so even the longest horizon is a short glimpse forward and not a route across the plaza.

The horizons were kept seperated on purpose because each one is its own trained model with its own checkpoint. Asking a network to commit to twenty metres is a different problem from asking it to commit to one. Reporting them as a single number would have hidden where the method holds and where it starts to fail.

The short horizons perform well. At H20 the average displacement error sits around 0.45 m, and the predicted endpoint lands inside a one metre tolerance about 83% of the time. For an architect that is a usable signal. It says the immediate intention of a pedestrian, the next step or two, is legible from recent motion alone. This agrees with what the earlier experiments had already shown, that recent movement is the strongest evidence of what happens next.

The error grows in a way that is easy to read. Average displacement rises to about 1.0 m at H60, 1.4 m at H100, 2.6 m at H200 and roughly 5.1 m at H400. Endpoint success falls along the same line, from 83% down to 55%. The final displacement error, measured only at the last predicted point, runs higher still, from about 0.7 m at H20 to 10 m at H400. That is the number I watch most, because the endpoint is where a predicted path either agrees with the plan or clearly does not. Small uncertainties at each step accumulate, and a model that is confident about the next metre has very little to hold onto twenty metres out.

What matters more than the size of the error is its shape. The predictions do not only drift. They shorten and straighten. At H100 the median predicted path is about 1.9 m long while the recorded path over the same window is about 5.3 m. The model reaches less far than the person actually walked, and it smooths the turns out of the way. Heading error stays high across every horizon, between about 74 and 84 degrees, which tells the same story from another angle. Direction is the hardest thing to get right, and it is the first thing to go.

This straightening is the same failure that appeared on the sandbox, now measured properly on real data, and the reason points to the fix. Trained to minimise error one step at a time, the model learns that the safest guess is the average of what usually comes next, and the average of many slightly different turns is close to a straight line. Two later experiments were built to push against this directly. The first, a magnitude aware objective, corrected the tendency to under-reach and brought the predicted path length back close to the real one. The second, a curvature aware objective, went after the straightening itself and recovered much of the shape, roughly a sixfold improvement in a macro shape measure, at a modest cost to directional accuracy. This curvature aware model is the final model, and it is the one used for the qualitative examples below. The progression from the baseline to the magnitude fix to the curvature fix is the geometry of the problem being corrected one property at a time, and each step was a controlled change to the training objective rather than a new architecture.

The architectural reading follows from this. A short prediction describes intention. A long prediction describes tendency. At one to five metres the predicted path is close enough to the real one that it can be laid over a plan and trusted as a local reading of movement. Past ten metres it should be treated as a soft field of likely direction, not a line that someone will follow. The clearest examples come from the open plazas, Catalunya, Espanya and the esplanade, where paths are long enough to read as a walk rather than a pause. In the tighter sites the recorded motion is shorter and the prediction has less to say.

To show this I use two kinds of figure, each from a different model. The per horizon error panels come from the baseline model, ranked by error so the best and worst cases at each horizon sit side by side. The curated best and worst examples, chosen for legibility rather than lowest error, come from the final curvature aware model. Using both is deliberate. The baseline shows the raw behaviour of the method. The final model shows how far a change in the training objective could push the geometry back toward the real path. Neither figure is standing in for the other, and I say which is which so the comparison is not misread as one model doing both.

[FIGURE 3.16 HERE]
Source file: THESIS/figures/booklet/horizon_rollouts/H100_best_track4703.png | THESIS/figures/booklet/horizon_rollouts/H100_worst_track3160.png
Proposed caption: Best and worst H100 rollouts, at roughly five metres. Purple marks the selected best case and green the selected worst; black is the observed history and the grey dotted line the recorded ground truth. Even the worst case stays a plausible walk rather than collapsing.

[FIGURE 3.17 HERE]
Source file: THESIS/figures/booklet/horizon_rollouts/H400_best_track112.png | THESIS/figures/booklet/horizon_rollouts/H400_worst_track540.png
Proposed caption: The same comparison at H400, roughly twenty metres, with purple best and green worst against the grey dotted ground truth. The best case still follows the shape of the walk; the worst drifts and straightens, the long-range limit the curvature aware objective was built to push back.

The stress test sits at the edge of what the data can support. Here the model is pushed well beyond the range it was trained and validated on, out to roughly 40, 50 and 60 metres, to see whether the predicted motion stays physically plausible or breaks down into noise. The point of the test is not accuracy, since there is no reliable ground truth that far out. It is a check on how the model fails. A prediction that keeps moving like a person, even when it is wrong, is more useful to a designer than one that falls apart.

One word near these results is easy to misread. The endpoint measure I quote is a tolerance on where a path ends, not a judgement of whether its whole shape was right. A prediction can land close to the true endpoint while taking a straighter route to get there. The two are related but not the same, and the distance between them is exactly the curvature problem the final model was built to close. Read together, the horizons give the thesis its most grounded claim. The method is reliable at the scale of a few steps, informative at the scale of a plaza crossing, and speculative beyond it.

<!-- TRANSITION -->

The horizons draw the boundary of the method. It is reliable at the scale of a few steps, informative across a plaza, and speculative beyond. That is a usable result for an architect, but only if it can be reached without a command line and a training run. A finding that lives inside a research pipeline reaches no one who actually designs. The last section of this chapter turns to the prototype, the attempt to wrap everything before it into something a designer could sit in front of.

<!-- /TRANSITION -->

## Building the prototype

The last piece of the work is not an experiment. It is an attempt to make everything before it usable by someone who is not running Python scripts. The prototype is a front end that wraps the pipeline into a single flow an architect could actually sit in front of.

The flow follows the pipeline. A user uploads footage of a space and its plan. They calibrate, matching points between the two so the movement can be placed in real coordinates. The system processes the footage, and the user inspects the result, both the observed movement, as trajectories and behavioural maps, and the predicted movement rolled forward by the model. The intent is to turn a research process that currently takes manual steps and command line tools into something closer to an application, where the analysis is a few clicks rather than a pipeline run.

[FIGURE 3.18 HERE]
Source file: THESIS/figures/booklet/platform_screenshots/initiation.png | THESIS/figures/booklet/platform_screenshots/calibration.png
Proposed caption: The opening and calibration screens. The landing page (left) introduces Motion Pixels as a tool for mapping spatial intelligence and starts a new study. Calibration (right) shows the camera view and the site plan side by side, where the user matches corresponding landmarks to establish the homography that links image to plan.

Each step in the interface mirrors a step in the research. The upload screen is where a space enters the system as nothing more than a video and a drawing. The calibration screen is where the two are locked together, and it is the step that still needs a human, because deciding which point in the footage matches which point on the plan is a judgement the system cannot yet make reliably on its own. Processing is where the pipeline runs, and inspection is where the result becomes something to read, switching between the observed trajectories, the behavioural maps and the predicted paths.

The inspection view is the part that matters most, because it is where the two halves of the research meet. Observed movement and predicted movement can be shown over the same plan, so a designer can compare what people did with what the model expects, and decide for themselves how far to trust the forecast. That decision, rather than a single accuracy figure, is what the tool is meant to support.

[FIGURE 3.19 HERE]
Source file: THESIS/figures/booklet/platform_screenshots/dashboard.png
Proposed caption: The studio dashboard for the Plaça Espanya demonstration. The space is drawn as layered spatial information, with speed, flow fields, bottlenecks and predictions switched on and the animation paused a few seconds into playback. The right panel holds saved studies, layer switches and export controls, and the timeline and prediction-horizon controls sit below the map. This is the bundled precomputed demonstration, not a newly processed dataset.

Its status matters, because a convincing interface can imply more than it delivers. The prototype is a demonstrator with mocked data. It shows the intended experience and the shape of the tool, but it is not connected to a live backend that runs detection, calibration and inference on demand. The movement and predictions it displays stand in for what a finished system would compute, not for a real time result. Presenting it as a working product would overstate where the research is.

Turning the demonstrator into a working tool is mostly engineering, but not entirely. The mocked parts are the ones that are genuinely hard, automatic calibration, reliable tracking on unseen footage, and inference fast enough to feel interactive. Each of those is a real problem in its own right, and pretending the interface has solved them would repeat exactly the kind of overstatement this thesis has tried to avoid. What the prototype does show is the destination, and a plausible path toward it.

An architect is not going to run a tracking model from a command line, and they should not have to. If observed and predicted movement are to become a normal part of how a space is evaluated, they have to arrive in a form that fits an architect's existing tools and habits, next to the plan, at the scale of a project, without a data science team in between. The prototype is a sketch of that form. It is included in the thesis because the research question was never only whether movement can be predicted, but whether the result can be made to matter to design.

Seen next to the maps and the predictions, the prototype closes the loop the first chapter opened. The drawing describes intention. The footage records behaviour. The pipeline turns that behaviour into a spatial layer, and the interface hands that layer back to the person making the drawing. Whether the prediction is accurate to one metre or five, the more important shift is that movement is no longer lost once a space is occupied. It has a place to go.

[FIGURE 3.20 HERE]
Source file: THESIS/figures/booklet/platform_screenshots/save.png
Proposed caption: Exporting the drawing. The dialog previews the current studio composition and offers a high-resolution PNG or an editable SVG of the layers, so the analysis can leave the tool as a drawing an architect can keep working on. The prototype runs on the bundled demonstration data rather than a live backend.

<!-- TRANSITION -->

This chapter has taken Motion Pixels from a single plaza to a dataset across Barcelona, through the failure that shaped it, the schema it settled on, the maps it produced, the predictions it can and cannot make, and the interface that carries them. The results are specific and bounded. What they mean, for the model, for the method, and for the larger question of movement as architectural evidence, has been left mostly implicit so far. The next chapter steps back from the machinery to read those results, to name the limits plainly, and to ask what the whole exercise was actually for.

<!-- /TRANSITION -->

---

*Target word count: 6,000. Actual word count: see chapter audit in 01_SOURCE_MAP.md. Transition paragraphs excluded. Locked passage: the Horizon Rollouts paragraphs 1 to 4 reproduce the author-locked calibration text verbatim.*
