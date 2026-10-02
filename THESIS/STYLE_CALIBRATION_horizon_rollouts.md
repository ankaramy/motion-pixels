<!-- STYLE CALIBRATION — Chapter 3, "Predictive Layer: Horizon Rollouts". Target ~1000 words.
Paragraphs 1-4 are the AUTHOR'S edited text and are LOCKED (do not rewrite, rephrase, or correct,
including spelling, unless the author explicitly asks). They are the primary style reference
(see THESIS/05_STYLE_GUIDE_FROM_AUTHOR.md). Paragraphs 5+ are the earlier generated draft, NOT yet
author-reviewed and NOT locked. -->

## Predictive Layer: Horizon Rollouts

<!-- ===== LOCKED (author-edited) — paragraphs 1-4 ===== -->

Once the dataset was stable, the real question was how far ahead the model could see. And that's exactly what a horizon is: It is how many steps into the future the model is asked to predict from a short slice of observed motion. I trained and evaluated five of them. In the notation I use through this chapter they are H20, H60, H100, H200 and H400, which correspond to roughly 1, 3, 5, 10 and 20 metres of travel. The numbers are frames, not minutes. A person at a normal pace would cover that ground in a few seconds, so even the longest horizon is a short glimpse forward and not a route across the plaza.

The horizons were kept seperated on purpose because each one is its own trained model with its own checkpoint. Asking a network to commit to twenty metres is a different problem from asking it to commit to one. Reporting them as a single number would have hidden where the method holds and where it starts to fail.

The short horizons perform well. At H20 the average displacement error sits around 0.45 m, and the predicted endpoint lands inside a one metre tolerance about 83% of the time. For an architect that is a usable signal. It says the immediate intention of a pedestrian, the next step or two, is legible from recent motion alone. This agrees with what the earlier experiments had already shown, that recent movement is the strongest evidence of what happens next.

The error grows in a way that is easy to read. Average displacement rises to about 1.0 m at H60, 1.4 m at H100, 2.6 m at H200 and roughly 5.1 m at H400. Endpoint success falls along the same line, from 83% down to 55%. The final displacement error, measured only at the last predicted point, runs higher still, from about 0.7 m at H20 to 10 m at H400. That is the number I watch most, because the endpoint is where a predicted path either agrees with the plan or clearly does not. Small uncertainties at each step accumulate, and a model that is confident about the next metre has very little to hold onto twenty metres out.

<!-- ===== NOT LOCKED — earlier generated draft, pending author review ===== -->

What matters more than the size of the error is its shape. The predictions do not only drift. They shorten and straighten. At H100 the median predicted path is about 1.9 m long while the recorded path over the same window is about 5.3 m. The model reaches less far than the person actually walked, and it smooths the turns out of the way. Heading error stays high across every horizon, between about 74 and 84 degrees, which tells the same story from another angle. Direction is the hardest thing to get right, and it is the first thing to go.

This is where the architectural reading has to stay honest rather than flattering. A short prediction describes intention. A long prediction describes tendency. At one to five metres the predicted path is close enough to the real one that it can be laid over a plan and trusted as a local reading of movement. Past ten metres it should be treated as a soft field of likely direction, not a line that someone will follow. I would rather say that plainly than present the long horizons as something they are not. The clearest examples come from the open plazas, Catalunya, Espanya and the esplanade, where paths are long enough to read as a walk rather than a pause. In the tighter sites the recorded motion is shorter and the prediction has less to say.

To show this I use two kinds of figure, and it is worth being clear about where each one comes from. The per horizon error panels come from the baseline model, MODEL_X, ranked by error so the best and worst cases at each horizon sit side by side. The curated best and worst examples, chosen for legibility rather than lowest error, come from the final model, MODEL_XC, which was trained with a curvature aware objective to recover some of the shape the baseline had lost. Using both is deliberate. The baseline shows the raw behaviour of the method. The final model shows how far a change in the training objective could push the geometry back toward the real path. Neither figure is standing in for the other, and I say which is which so the comparison is not misread as one model doing both.

The stress test sits at the edge of what the data can support. Here the model is pushed well beyond the range it was trained and validated on, to see whether the predicted motion stays physically plausible or breaks down into noise. [AUTHOR INPUT REQUIRED: the outline specifies a 40, 60 and 100 m stress test; the verified stress runs in the repository extend to roughly 40, 50 and 60 m. Confirm the intended ranges, or the outputs to report.] The point of the test is not accuracy, since there is no reliable ground truth that far out. It is a check on how the model fails. A prediction that keeps moving like a person, even when it is wrong, is more useful to a designer than one that falls apart.

One word near these results needs care. The endpoint measure I quote is a tolerance on where a path ends, not a judgement of whether its whole shape was right. A prediction can land close to the true endpoint while taking a straighter route to get there. The two are related but not the same, and the distance between them is exactly the curvature problem the final model was built to close.

Read together, the horizons give the thesis its most grounded claim. The method is reliable at the scale of a few steps, informative at the scale of a plaza crossing, and speculative beyond it. It is a modest result. It is also a true one, and it is the kind of result an architect can actually build on.
