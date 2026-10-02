# Conclusion

This thesis began with a gap that every architect knows and few can measure. A drawing describes how a space is meant to be used. The people who use it answer with their own movement, and that answer is usually lost once the building is occupied. Motion Pixels was an attempt to hold onto it, to turn ordinary video of a public space into movement that can be measured, placed back onto the plan, and read as evidence.

The path there was not straight. It started on one plaza in front of MACBA, where the first models failed in a way that turned out to be useful. They flattened the curved, turning movement that made the site worth filming, and chasing that failure is what shaped the rest of the work. A capacity test showed the architecture could represent angular movement once it had enough examples, which redirected the research from fixing the model to feeding it. The dataset grew across Barcelona, and the prediction problem was split into horizons so that the method could be judged at each range rather than as a single figure.

What that judgement returned is a bounded claim, and I have tried to keep it bounded throughout. Short-range prediction works. At the scale of a step or two, movement follows reliably from recent motion, and spatial context adds to that once there is enough data to learn from. As the horizon grows the prediction weakens, straightening and falling short, until beyond about five metres it is better read as a direction than a route. Later experiments that changed the training objective recovered part of the missing shape and confirmed that the limits were as much about how the model learned as about how much it saw. None of the results transfer to a genuinely unseen space, and the thesis does not pretend they do.

Set against the question the work opened with, this is enough. The question was whether pedestrian movement carries enough structure to be treated as spatial evidence, and the answer is a qualified yes. It carries that structure most clearly at the scale where an architect actually reasons, at a threshold, an edge, a crossing. Prediction was the test of that structure, not the product. The product, if there is one, is the shift in what an architect can hold. Movement stops being anecdote and becomes a layer that sits next to the plan, described in terms a designer already uses.

Much is still manual, still small, still tied to five spaces in one city. The calibration is drawn by hand, the dataset is modest, and the long-range predictions are weak. These are real limits, and I have kept them visible rather than let a convincing map or a clean interface imply more than the work has earned. A partial tool that is clear about its edges is more useful to a designer than a confident one that is not.

I could keep going on the prediction itself, on ways to shave the error down or push the accuracy toward the centimetre. But the larger actor in this thesis was never the model. It is spatial design. Motion Pixels is an attempt to read the feedback between behaviour and architecture, the quiet back and forth in which a space shapes how people move and their movement, in turn, reveals what the space is actually doing. Behaviour is not a by-product of architecture to be tidied away once a building opens. It is information, and it can feed back into design.

If the work that follows this thesis widens the dataset, sharpens the prediction, and lifts the reading into three dimensions, the point of it will not be a better forecast. It will be a better understanding of the spaces we design, and a slow move toward something worth calling spatial intelligence.

---

*Target word count: 600. Actual word count: see chapter audit in 01_SOURCE_MAP.md.*
