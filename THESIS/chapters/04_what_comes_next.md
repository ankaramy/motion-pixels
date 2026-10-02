# Chapter 4. What comes next?

The previous chapter followed the research from a single plaza to a dataset spread across Barcelona, and from a first failure to a set of measured results. This chapter steps back from the machinery and asks what the whole exercise actually established, where it stops, and what it points toward. The tone shifts here on purpose. The earlier chapters reported. This one interprets, and it tries to keep interpretation separate from the results that support it.

## What did the model learn

The clearest finding is also the simplest. Recent movement is the strongest predictor of short-term motion. Across every experiment, a person's next step or two followed most reliably from how they were already moving, and this held before any spatial feature was added. If Motion Pixels shows one thing without qualification, it is that the immediate future of a pedestrian is written mostly in their recent past.

The horizon results give that claim a scale. At the shortest range, a metre or so, the prediction is close enough to be trusted as a local reading. By about five metres it still holds together as a plausible path. Beyond ten it becomes a direction of travel rather than a route. So the model did not learn to see the future in general. It learned to extend the present a short way, and the useful part of that extension happens to fall at the scale where an architect reasons about a threshold or a turn.

Spatial context helps, but conditionally. The features describing where a person is and what lies around them improved prediction when there was enough data to learn from, and did little when there was not. On the small sandbox, adding spatial information barely moved the results. On the larger dataset it earned its place. The space does leave a mark on movement, but reading that mark takes more examples than reading motion alone. Recent motion is present in every trajectory, while the effect of a particular corner or edge only becomes legible once the model has seen many people meet it.

There was also a lesson that had nothing to do with prediction. The behavioural maps, built only from observed movement, already surfaced things a plan does not show, where a plaza slows people down, where crossings concentrate, where a space empties out. Before any forecast, describing the movement well turned out to be worth a great deal on its own. Not all of the value here is in the model.

Direction and curvature are harder than position, and this ran through the entire project. A model could place a person roughly where they were going while getting the shape of the path wrong, because it flattened turns into straight lines. There is a real distinction hidden in that sentence. Landing near the right endpoint is not the same as tracing the right path, and a prediction can do the first while failing the second. Predicting that someone will be five metres ahead is easier than predicting that they will arrive there along a curve. The distinction matters to an architect, because the curve is often where the behaviour is, the swerve around an obstacle or the arc toward an entrance.

The model also under-reached. Left to minimise error one step at a time, it predicted paths that were not only too straight but too short, stopping before the person did. Length and shape were two separate things to get wrong, and the later objective changes had to correct them one at a time rather than together.

The last finding is the one that most shaped the research. The early failure at MACBA turned out to be primarily a data problem rather than a flaw in the model. The architecture could represent angular movement once it had enough examples, which is what the capacity test showed. I described this at the time as the system being data-bound rather than architecture-bound, and that is still the right first reading. It is not the whole story. The later experiments that changed the training objective, not the data, also recovered a large part of the missing shape. So the fuller version is narrower. A shortage of data was the main early obstacle, and the way the model was trained was a second one. Neither alone explains the collapse, and it would be too neat to blame only the dataset.

Put together, these findings sketch a boundary rather than a headline. Movement is partly legible. It is most legible at short range, more legible in motion than in geometry, and legible in space only with enough examples. That boundary, and where it falls, is the real result.

## Limitations of research

The results come with real limits, and naming them precisely is more useful than a general disclaimer.

The dataset is small and narrow. Three and a half thousand trajectories from five recordings is not a large or diverse sample of how people move through cities. The sites were chosen for variety, but they are still five spaces in one city, filmed under particular conditions on particular days. Any claim the research makes should be read against that scale, and I have tried to keep the claims proportionate to it.

The movement itself is imperfect. Detection and tracking fail in ordinary ways. People are occluded by others, identities are occasionally dropped or swapped between two nearby pedestrians, and a jittery track adds noise that looks like real motion. Much of this washes out across thousands of trajectories, but it sets a floor under how clean the data can be, and it shows when a single path looks strange. Some of the sharpest turns in the raw data are not people at all. They are the tracker changing its mind.

There is also the matter of what the camera saw. A single viewpoint captures only part of a space, and people appear and disappear at the edges of the frame, so a trajectory is only ever the portion of a journey that fell within view. The dataset is a record of movement through a camera's field, not through the whole space, and that framing quietly shapes everything downstream.

Calibration and spatial encoding still depend on manual work. For each recording, points were matched between footage and plan by hand, and the walkable areas and obstacles were drawn by hand. This is slow, it does not scale, and it is one of the clearest gaps between the current method and a tool that anyone could pick up. It also means a person's judgement enters the data at the calibration stage, which is not necessarily wrong but should be on the record.

Long predictions degrade, and they degrade in a particular way rather than at random. As the horizon grows, error accumulates and the predicted path tends to straighten and fall short of the real one. The objective changes in the final experiments reduced this without removing it. Beyond roughly five metres, a prediction should be read as a likely direction rather than a committed route.

The stress test pushed the model far past its validated range, and its outputs there are exploratory. They show that the prediction stays physically plausible rather than dissolving into noise, but there is no ground truth at that distance to check them against, and they should not be read as accuracy. The behavioural maps carry a related caution. A warm cell on a density map marks where movement concentrated, not a proven circulation failure. The map raises the question. A designer still has to answer it.

Two limitations concern how the results are read rather than how they were produced. The first is that the endpoint measures used through the horizon results describe where a path ends, not whether its whole shape was right, so a good endpoint number can sit over a path that took a straighter route than the real one. The second is the split. Every model here was tested on people within the same sites it was trained on, never on a genuinely unseen space. The capacity test, which duplicated data to probe what the architecture could fit, is memorisation and says nothing about a new place either. Whether the method transfers to a space it has never seen is a question this work sets up and does not answer.

[FIGURE 4.1 HERE]
Source file: THESIS/figures/booklet/limitations.png
Proposed caption: The limitations of the current method at a glance, from the size and narrowness of the dataset and the noise in tracking to the manual calibration and the decay of prediction over long horizons. Each is a real constraint, and each points to a direction for further work.

## The meaning of it all

It would be easy to read this thesis as a project about prediction, and to judge it by how accurate the predictions are. That would miss the point.

Prediction was never the final objective. It was the test. The question underneath was whether movement carries enough structure to be treated as spatial evidence, and prediction was the sharpest way to probe it. The fact that short-range prediction works at all, and that spatial context measurably helps once the data supports it, is the finding that matters, more than any single error figure. A weaker result at long range does not undo a solid one at short range. It just marks where the evidence thins.

What the work is really proposing is that observed and predicted movement can become another layer of architectural information. A plan records intention. Movement records use. For most of architectural practice the second of these is lost once a building is occupied, surviving as memory and anecdote rather than as anything an architect can hold and compare. Motion Pixels is an argument that it does not have to be lost, that it can be captured, measured, placed back onto the plan, and read next to it.

To make that concrete: an architect designing a new plaza could load footage of an existing one that works in a similar way, read how people actually move through it, and carry that reading into the new design as evidence rather than assumption. The comparison is not a guarantee, and no two spaces are identical, but it replaces a guess with an observation, which is most of what evidence ever does in design.

This is where the two references from the first chapter meet. Space Syntax reasons from configuration toward likely movement, and Whyte reasoned from observed behaviour back toward the space. This project sits between them, taking Whyte's starting point in the footage and trying to return the result to the measured, plan-based world Space Syntax works in. The contribution is less a new model than a way of moving between the two.

That reframes the ambition. The goal is not to tell an architect where a crowd will go and have them design to it. It is to give them a way to see how a space is actually used, to compare that with how it was meant to be used, and to let the difference inform the next decision. The sequence is modest. Observe behaviour, use it to understand a space more truthfully, and let that understanding feed design.

Read this way, the limitations above are less damaging than they first appear. A method that gives a trustworthy reading at the scale of a few metres, and a weaker one beyond, is still useful to a designer trying to understand a threshold, an edge, or a bottleneck. The value is in the reading, not in the range.

A partial tool that is clear about its limits is more useful than a confident one that hides them. An architect can work with a reading that says trust me to five metres and treat the rest as tendency. They cannot work with a black box that claims more than it can deliver. The bounded, calibrated nature of the result is not a weakness to apologise for. It is part of what makes it usable.

## Future research directions

Several directions follow directly from the limits, and I list them in roughly the order I would pursue them.

The most immediate is more data, across more sites and more kinds of space. The single clearest constraint on this work was the size and narrowness of the dataset, and almost every result would be firmer with a larger, more varied one. This is also what a real test of transfer to unseen sites requires, since that test only means something once there are enough sites to hold some back. More data would also let the dataset be balanced deliberately, so that constrained spaces like stairs are not drowned out by open plazas, as they were here.

Close behind is removing the manual bottleneck. Calibration and masking are the steps that stop this from scaling, and automating even part of them, so that a new recording could be prepared without hours of hand work, would do more for the method's reach than any change to the model. It is less glamorous than a new architecture and probably more important.

The prediction itself has room to improve on the properties that proved hardest. Heading and curvature, and the behaviour of the model at long horizons, are where the objective-based experiments already pointed, and that line of work is not exhausted. The aim further out is a more developed and better tuned prediction tool, trained on a far larger dataset, dependable enough to be pointed at a new space and trusted for the near-future reading it returns. The model also predicts each person on their own, as if they moved through an empty space. Crowds are not made of independent individuals, and adding pedestrian-to-pedestrian interaction, which much of the literature already treats, would bring it closer to how movement works when a space is busy.

The behavioural maps could also become richer. At the moment each one reads a single feature, flow, speed, or density. Maps that combine several of these, or that surface more of the information the encoding already holds, would give a fuller picture of how a space is used, closer to a designer reading than a single metric.

The larger step is dimensional. The current features reduce a space to distances to obstacles and boundaries in two dimensions, and the maps are drawn flat on the plan. A natural extension is to reconstruct the space itself in three dimensions from the same video, and to place the behaviour back into that model, so the heatmaps, flows and predicted paths can be read as volumes in three dimensions rather than as marks on a flat drawing. There is also a temporal dimension the current work barely touches, since a single recording cannot show how a space changes across a day, a season, or a use.

The furthest ambition is a loop rather than a pipeline. Everything here runs in one direction, from footage to prediction. The more interesting version would close that loop, so a designer could observe a space, predict how a change might affect movement, modify the design, and evaluate the result, then go round again. That is a long way from where this thesis stops. It is the destination that gives the work its point, a behaviour-informed way of designing rather than only a behaviour-reading one.

Two quieter directions would strengthen the foundation under all of this. One is validation, checking the predicted and observed readings against the judgement of people who know a space well, so that the maps and forecasts earn trust beyond their error figures. The other is integration. Motion Pixels does not need to stand alone, and it would be stronger sitting alongside the tools architects already use, feeding observed movement into a Space Syntax reading or a building model rather than competing with them.

<!-- TRANSITION -->

This chapter has read the results rather than reported them. Recent movement carries most of the short-range signal, space helps once there is enough data to learn it, direction is the hardest thing to hold, and the method is bounded in a way that is clear about where it can be trusted. The future directions follow from those limits rather than from ambition alone. What remains is to draw the argument together, to say what Motion Pixels set out to test and what it found, and to place a small but real result back in front of the architect it was meant for. That is the work of the conclusion.

<!-- /TRANSITION -->

---

*Target word count: 2,500. Actual word count: see chapter audit in 01_SOURCE_MAP.md. Transition paragraphs excluded.*
