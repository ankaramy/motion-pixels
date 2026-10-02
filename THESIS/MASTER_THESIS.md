# Motion Pixels

## Mapping Out Spatial Intelligence

**Author:** Ramy Anka

**Advisor:** Professor Wassim Jabi

**Institution:** Institute for Advanced Architecture of Catalonia

**Master programme:** MaAI, Master in AI for Architecture and the Built Environment

**Thesis cluster:** Spatial Intelligence, AI for Perception, Movement, Typology and Urban Performance

**Academic cycle:** 2025-2026

**Location and date:** Barcelona, June 2026

# Abstract

An architectural drawing describes how a space is meant to be used. The people who use it move in ways the drawing never records, and once a space is occupied that behaviour is usually lost. Motion Pixels asks whether it can be kept. The thesis investigates whether pedestrian trajectories can be predicted from the relationship between human behaviour and architectural space, and whether the result can become a layer of evidence an architect can read alongside the plan.

The method turns ordinary video into measured movement. Pedestrians are detected and tracked, the footage is calibrated to the architectural plan by homography so that movement is expressed in real metres, and each trajectory is encoded in terms of both its motion and its spatial situation, its distance to obstacles and boundaries. A recurrent model, a Long Short-Term Memory network chosen for its stability over long rollouts, then predicts movement forward from a short window of observed motion. The work was developed first on a single sandbox site, the esplanade in front of MACBA in Barcelona, and then on a dataset of five calibrated recordings across the city containing 3,534 tracked trajectories.

An early failure shaped the research. The first models flattened the curved movement the sandbox was chosen to capture. A controlled capacity test showed that the architecture could represent angular movement once it had enough examples, which identified the problem as primarily a shortage of data rather than a flawed design. Prediction was then evaluated across several horizons, from roughly one to twenty metres of travel.

The results are bounded and consistent. Short-range prediction is reliable, with an average displacement error near half a metre at the shortest horizon and endpoint placement inside a one metre tolerance about eighty-three percent of the time. Accuracy falls as the horizon grows, and predictions tend to straighten and fall short of the real path. Later experiments that changed the training objective recovered a large part of the missing shape, which confirmed that the limits were as much about how the model learned as about how much data it had. The models were evaluated on unseen people within the same sites, not on entirely unseen spaces, and no claim of transfer beyond the studied sites is made.

The contribution is less a model than a way of working. Prediction is used as a test of whether movement carries usable spatial structure, and it does, most clearly at the scale where an architect reasons about a threshold, an edge, or a crossing. Observed and predicted movement, together with behavioural maps of flow, speed and density, form an additional layer of architectural information that returns lived behaviour to the design process rather than leaving it in memory.

**Keywords:** pedestrian movement, trajectory prediction, behavioural mapping, architectural analysis, computer vision

# Preface

This book follows the research in the order it happened. The first chapter sets out the problem, the gap between how a space is designed and how it is used. The second reviews the work Motion Pixels builds on and describes the pipeline that turns video into data. The third is the longest, and it moves from a single test site to a dataset across Barcelona, through the experiments, the maps, the horizon predictions and the prototype. The fourth interprets the results and looks ahead. A reader who wants the argument without the machinery can read the first and last chapters and the conclusion. A reader who wants the evidence will find it in the third.

The intention throughout has been to keep claims proportionate to what the work actually shows. Where a result is strong it is stated plainly, and where it is weak or unfinished it is marked as such rather than smoothed over.

## Acknowledgments

This thesis was done over the course of a single year, and it has been an adventure filled with learning and self progression. Professor Wassim Jabi was my thesis advisor, and I learned a great deal from him.

I am grateful to Angelos Chronis and Areti Markopoulou for initiating the programme and for offering me a full scholarship, and to Eleni Karafylli, the programme coordinator, for her constant support since the first year.

I thank my parents, Milad and Marie, and my sister Zeina, for helping me finance my studies. And I thank Elias and Evangelo for their unwavering support and their moral and technical help during the thesis.

## Declaration of AI Use

Agentic coding tools were used in the creation of the code for the computational pipeline. Large language models were used as spell-checking tools for the writing of this thesis and in the creation of the PDF for this booklet.

# Index


# Chapter 1. The flaw in the plan

## Introduction to topic

An architectural drawing is a description of intention. It says where a wall should stand, how wide a passage should be, where a person is expected to enter and where they are meant to arrive. The drawing is confident about all of this. What it cannot describe is what people actually do once the building or the plaza is finished and occupied.

I keep returning to a simple observation. No matter how carefully we design for a client, and no matter how well we think we understand the people who will use a space, one instinct always survives the drawing. A person will take the path they want, not the path we drew for them. The designed route and the desired route are rarely the same line.

This is not a failure of design. It is the normal condition of architecture. A plan proposes an order, and the people who move through it answer with their own. The relationship between the two is a kind of negotiation that goes on quietly, every day, in every public space. A doorway invites, a corner slows people down, a shortcut appears across a lawn that was never meant to be crossed. The building sets terms and the crowd renegotiates them.

The evidence of this negotiation is everywhere, but it is rarely collected. A worn strip of grass records a shortcut more honestly than any survey. A cluster of people always forming at one end of a plaza says something the plan did not predict. These are readings of a space that only exist once it is in use, and they tend to disappear from the architectural record precisely because they arrive after the drawing is finished. The plan is archived. The behaviour is forgotten.

Architects have always known this relationship exists. The problem is that we have mostly known it as intuition. We talk about how a space wants to be used, about desire lines, about places that feel alive and places that feel dead. The vocabulary is rich and the observation is real, but it stays largely qualitative. The link between a spatial configuration and the behaviour it produces has been described and theorised for decades. It has been harder to hold it as evidence, in numbers, tied to a specific place.

There have been serious attempts to close that gap, and I return to them in the next chapter. What most of them share is a starting point in the space itself, in the geometry of the plan, rather than in the movement of the people. My interest runs the other way. I wanted to begin with the behaviour, with the actual traces people leave as they cross a real site, and then ask what that behaviour reveals about the space.

That ambition is broad enough to become vague if it is not anchored. Movement in public space is shaped by weather, habit, time of day, the company a person keeps, and a hundred things a drawing will never contain. If the research tried to account for all of it at once, it would produce a description of everything and an explanation of nothing. So the work needed a narrow, concrete case to build a method around before it could say anything general. It needed a site, a way of watching it, and a way of turning what was watched into something an architect could read.

Motion Pixels grew out of that need. The idea is direct. Ordinary video of a public space already contains a detailed record of how people move through it. If that record can be pulled out of the footage, placed back onto the architectural plan in real measurements, and described in terms an architect recognises, then movement stops being an anecdote. It becomes a layer of spatial information that sits next to the plan rather than vanishing once the building is occupied.

Video is the right medium for this because it is ordinary. Most public space is already filmed, and a single camera looking at a plaza holds more behavioural detail than a team of observers could record by hand. What has been missing is not the footage but a reliable way to convert it into spatial measurement. Progress in detection and tracking has made that conversion possible at a scale that was not practical when earlier researchers were counting people by eye.

There is a difference between describing where people have been and understanding why they went there. A heatmap of a plaza is already useful, but it is a record of the past. It shows the result of the negotiation without explaining its terms. If movement genuinely follows from the relationship between a person's recent motion and the space around them, then that relationship should be strong enough to do more than describe. It should be able to anticipate, at least a little. Prediction, in this thesis, is less a product than a test of how much structure the behaviour really holds.

This chapter sets up the problem the rest of the thesis works on. Before the tools and the experiments, the claim itself needs stating. It is not that movement can be reduced to a formula, or that behaviour can be designed with certainty. It is narrower and, I think, more useful. Observed movement carries structure, that structure is legible, and reading it can change how a designer understands a space they thought they already knew.

[FIGURE 1.1 HERE]
Source file: THESIS/figures/booklet/beginning_1.jpg | THESIS/figures/booklet/beginning_3.jpg | THESIS/figures/booklet/beginning_4.jpg
Proposed caption: Desire lines worn into planted ground, where people ignore the paved routes laid out for them and cut the path they actually want. The designed path and the desired path rarely coincide, and the gap between them is what this thesis tries to read. Stock photographs.

## Early references

Two bodies of work stand behind this starting point, and they approach it from opposite ends.

The first is Space Syntax, developed by Bill Hillier and his colleagues at University College London. Their argument, set out in *The Social Logic of Space* and later in *Space is the Machine*, is that the layout of a space is not a neutral container for social life but an active part of it (Hillier and Hanson, 1984; Hillier, 1996). By describing a plan as a network of connected spaces and measuring properties such as how integrated or how segregated each part is, Space Syntax showed that certain configurational measures correlate with observed patterns of movement and co-presence. Hillier called part of this natural movement, the idea that much of where people go is set by the structure of the street network itself, before any particular shop or attraction is added (Hillier, 1996). A well integrated street tends to carry more movement simply because of where it sits in the grid.

This mattered to me for two reasons. It established that the relationship between configuration and behaviour can be measured rather than only felt. And it located that measurement firmly in the plan, in the geometry, which is exactly the side of the relationship I wanted to complement rather than repeat.

The second reference is William H. Whyte. Where Space Syntax reasons from the plan, Whyte reasoned from the pavement. In *The Social Life of Small Urban Spaces* he and his team filmed New York plazas over long periods and simply watched what people did (Whyte, 1980). The findings were often plain to the point of being funny. People sit where there is something to sit on. The most popular spaces were not the grand ones but the ones that offered small, ordinary comforts at a human scale. His team noticed that movable chairs let people arrange themselves and claim a spot, and that sun, water, food and the presence of other people drew crowds more reliably than open symbolic space. Whyte is often cited for the observation that generous, sittable edges do more for a plaza than almost any monumental gesture.

[FIGURE 1.2 HERE]
Source file: THESIS/figures/booklet/whyte.jpg | THESIS/figures/booklet/whyte_2.jpg
Proposed caption: William H. Whyte filming and observing New York plazas for The Social Life of Small Urban Spaces. He treated patient observation of real behaviour as architectural evidence, and used film to capture it over time. Images courtesy of the Project for Public Spaces.

What I take from Whyte is method as much as message. He treated observation as a legitimate form of architectural evidence, and he used film to do it, because film captures behaviour over time in a way a survey cannot. Motion Pixels is an attempt to give that patient watching a contemporary set of instruments. The camera is still doing the observing. What has changed is that the movement in the footage can now be extracted, measured, and placed back onto the plan automatically, at a scale of thousands of trajectories rather than a clipboard of counts.

Between these two references sits the position this thesis takes. Space Syntax reads behaviour from the space. Whyte reads the space from behaviour. Motion Pixels leans toward Whyte's direction, starts from the observed movement, but tries to bring the result back into the measured, plan based world that Space Syntax works in.

## Research Question

If configuration and behaviour are genuinely coupled, then that coupling should leave a trace, and a trace can in principle be learned. When the coupling is weak, or when a design ignores it, the symptoms are familiar to anyone who has watched a space fail. Signals that point the wrong way. People hesitating at a junction that made sense on paper. Crowds thickening into a bottleneck where two flows were never meant to meet.

These are not exotic problems. A station concourse where the signage points one way and the crowd insists on another. A plaza with a handsome diagonal that no one uses because the entrances sit at its corners. Failures like these are ordinary, and they are expensive, because they usually surface after construction, when the only remaining moves are barriers, signs and apology. If some of that behaviour could be read earlier, from footage of comparable spaces, the conversation could happen while it is still cheap to change the drawing.

The thesis takes that observation and turns it into a question that can actually be tested with data.

**Can pedestrian trajectories be predicted from the relationship between human behaviour and architectural and urban space?**

The word predicted is deliberate, and it needs a qualification. Prediction here is a way of testing whether the coupling carries real, usable information, not a promise that future movement can be known. If a model can look at a short stretch of someone's motion in a particular place and anticipate where they go next, then the relationship between behaviour and space contains structure a machine can pick up. If it cannot, that is also a finding. The question is a probe into how much of movement is legible, and at what range.

[FIGURE 1.3 HERE]
Source file: THESIS/figures/booklet/research_question.png
Proposed caption: The research question, framed as a test of whether the link between behaviour and space carries enough structure to anticipate movement.

## Hypothesis

The hypothesis behind the whole project is short.

Space shapes how people behave in it. If that behaviour can be captured, it becomes data. And if the data holds enough of the underlying structure, some of that behaviour can be anticipated rather than only recorded.

Stated that plainly, each part is a step the research has to earn. Capturing behaviour means extracting reliable movement from ordinary video and placing it correctly in space. Turning it into data means describing each trajectory in terms that carry both motion and spatial context. Anticipating it means asking a model to predict, and then measuring how far that prediction holds before it breaks down.

None of these steps is guaranteed. Extraction can fail on a crowded frame. A trajectory can be measured cleanly and still carry almost no predictive signal. A model can fit the past perfectly and generalise to nothing. The chapters that follow work through these steps in order, and they report where each one held and where it did not. This chapter only needs to establish why the question is worth asking, and why an architect, and not only a computer scientist, should be the one asking it.

[FIGURE 1.4 HERE]
Source file: THESIS/figures/booklet/hypothesis.png
Proposed caption: The hypothesis as a chain of claims the research must earn in turn, from capturing movement to anticipating it.

<!-- TRANSITION -->

This chapter has argued for a way of seeing before it has built anything. The plan states intention, movement records use, and the two rarely fall on the same line. The research question asks whether that movement can be predicted from its relationship to the space, and the hypothesis breaks that into steps the work still has to earn. None of it means anything without instruments. Before movement can be read as evidence it has to be pulled out of video, placed on the plan, and described in terms a model can learn from, and that machinery has to be justified against what already exists. The next chapter turns to those instruments, the research and the software this project builds on, and sets out the pipeline that turns a recording into data.

<!-- /TRANSITION -->

# Chapter 2. Tools and instruments for behavioral analysis

The previous chapter argued that observed movement can be treated as architectural evidence, and it framed that argument as a question about prediction. Before any of that can be tested, the research has to stand on existing work. Other people have studied how space shapes movement, how movement can be modelled, and how software already tries to simulate crowds. This chapter sets Motion Pixels against that background, and then describes the pipeline that turns a video into data. It ends with the question that any project handling footage of real people has to answer.

## Space Syntax

Space Syntax was introduced in the previous chapter as one of the two references the project reasons from. It deserves a closer look, because it is the clearest existing attempt to make the relationship between configuration and behaviour measurable.

The core move in Hillier's work is to stop treating a plan as a picture and start treating it as a network (Hillier and Hanson, 1984). A space is broken into its component parts, the lines of sight or movement that connect them are recorded, and the resulting graph is analysed. From that graph come measures such as integration, which describes how easily one part of the system can be reached from all the others. The finding that gave the method its weight is that these measures correlate with real patterns of use. More integrated streets tend to carry more movement, and they tend to do so whether or not there is an obvious destination on them. Hillier called this natural movement and treated it as evidence that the grid itself, not only its attractors, organises where people go (Hillier, 1996).

The analysis is not limited to axial lines. Related techniques describe what can be seen from a given point, the isovist, and build visibility graphs that measure how much of a space is exposed to each location. These give a reading of how open or enclosed a position feels, which is close to the kind of spatial context this project later encodes for each pedestrian. The vocabulary is different, but the instinct is shared. Both treat position within a configuration as something that can be quantified rather than only sensed.

The method has been applied widely, from single buildings to whole cities, and it has become a standard tool for reasoning about accessibility and co-presence. Its strength is also its boundary. Space Syntax reasons from the configuration outward to predicted movement potential. It describes what a layout affords. It says less about the specific, moment to moment behaviour of an individual crossing a particular plaza on a particular afternoon, because that was never what it set out to describe. A configurational measure can tell you a street should be busy. It cannot tell you how a given person will move along it in the next few seconds.

This is where Motion Pixels takes a different position rather than a better one. Space Syntax starts from the plan and derives likely movement. This project starts from the observed movement and works back toward the space. The two are complementary. One gives a configurational expectation, the other gives a behavioural record, and the interesting ground is where they can be compared.

[FIGURE 2.1 HERE]
Source file: THESIS/figures/booklet/space_syntax.jpg
Proposed caption: A Space Syntax reading of an urban grid, where the configuration of the network is used to estimate where movement should concentrate. It reasons from the drawing toward behaviour, the opposite direction to the observed record Motion Pixels builds. After B. Hillier, Space is the Machine.

## Literature review

The second body of work sits in computer vision and machine learning, where pedestrian trajectory prediction has been an active problem for years. The models matter to this thesis less as engineering and more as a map of what has already been tried, and on what kind of data.

Early learned approaches treated a trajectory as a sequence and used recurrent networks to continue it. A recurrent model reads a person's recent positions one step at a time, keeps an internal memory of the motion so far, and uses it to predict the next step. This is a natural fit for movement, because where someone goes next depends heavily on where they were just heading. The idea that reshaped the field was to let people influence one another. Alahi and colleagues introduced Social-LSTM, which gave each pedestrian their own recurrent network and then pooled the hidden states of nearby people into a shared social tensor, so that a prediction accounts for the neighbours crowding a person's path and not only their own history (Alahi et al., 2016). It was the first widely adopted way to treat a crowd as something more than a set of independent walkers, and almost everything that followed is a response to it.

The weakness of a single predicted line is that people rarely have one available future. At a junction a person might go left or right with almost equal reason, and a model that averages those options produces a path down the middle that no one would actually take. Generative models were introduced to represent that spread. Gupta and colleagues built Social GAN, which pairs a recurrent generator with a discriminator trained to tell real trajectories from predicted ones, and samples several socially plausible futures instead of committing to one (Gupta et al., 2018). Sadeghian and colleagues extended this with SoPhie, which adds two kinds of attention, one over the physical scene taken from an image of the space and one over the surrounding agents, so that the prediction respects both the built environment and the crowd (Sadeghian et al., 2019). SoPhie is the closest of these early models to the concern of this thesis, because it treats the scene itself, and not only the other people, as information that shapes where someone goes.

Attention then became the organising idea of the field. Giuliari and colleagues showed that a plain transformer, with no social pooling at all, could match or beat the more elaborate recurrent models on the standard benchmarks simply by attending over a person's own past positions (Giuliari et al., 2020). This was a useful and slightly deflating result, because it suggested that a large part of the benchmark score comes from modelling individual motion well rather than from modelling interaction. Later transformer work put the interaction back in a more principled way. Yuan and colleagues built AgentFormer, which attends jointly over time and over agents, so that the model can reason about how one person's move at one moment affects another person a few steps later, rather than handling the social and temporal dimensions separately (Yuan et al., 2021). Alongside this, conditional variational autoencoders became a common way to produce a distribution of futures, and graph based models described a crowd as a set of connected nodes and reasoned over that structure directly.

Not every useful idea is recurrent or attention based. Bai and colleagues argued that temporal convolutional networks, which read a sequence with stacked dilated convolutions rather than a recurrent loop, often match or exceed recurrent models on sequence tasks while being easier to train and more stable over long outputs (Bai et al., 2018). That property matters here, because stability over a long rolled-out prediction, rather than accuracy on a single next step, is exactly what this project cares about, and it is why a temporal convolutional network was one of the architectures tested before the final choice was made.

A smaller strand of work sits closer to architecture and planning. Some studies predict occupancy or flow at the level of a building or a zone rather than tracing individual paths, which is useful for sizing circulation and services but does not give the fine-grained movement this project needs. Others, in the architectural research community, use motion capture and spatial analysis to study how people occupy designed space, but they stop short of prediction. Motion Pixels sits between these, borrowing the pedestrian-trajectory machinery from the first group and the architectural framing from the second.

The field shares a common yardstick, and this thesis uses it too. Most of this work is judged by average displacement error and final displacement error, the mean distance between the predicted path and the real one and the distance at its endpoint. These two numbers recur throughout the literature. This thesis adopts them as well, but reads them as an architect rather than as a leaderboard, asking what a given error means for trusting a predicted path laid over a plan.

Two things stand out when this body of work is read together. The first is the data. Almost all of it is trained and evaluated on the same small set of public benchmarks, chiefly the ETH and UCY pedestrian videos and the Stanford Drone Dataset. These are valuable precisely because everyone uses them, which makes results comparable from one paper to the next. They are also not architectural. They are generic pedestrian scenes recorded for the purpose of studying pedestrians, not specific designed spaces tied to a plan an architect would recognise, and the scene, when it is used at all, enters as a background image rather than as a calibrated drawing. A model that scores well on them has learned to continue human movement in the abstract, not to read a particular place.

The second is the question being asked. Almost none of this work is framed the way an architect would frame it. The goal is a lower error on the shared benchmark, and the space is treated as context for the people rather than as the object of study. That is a reasonable choice for a machine learning field, and it has produced genuinely strong models, but it leaves the architectural question untouched. Motion Pixels borrows this machinery, and settles on a recurrent model in the end for reasons the next chapter sets out, but it points the machinery at a different target. The people are the instrument. The space is what the research is trying to read.

## Practical review

Alongside the research literature there is a mature software industry aimed at the same broad problem, and it already does parts of this well.

Bentley's tools, including OpenPaths and the LEGION product line, are used to model crowd movement in complex environments such as stations, stadiums and airports. They rely largely on agent based simulation, where many simulated pedestrians follow rules of movement and interaction, and the designer studies the emergent flow, capacity and congestion. Autodesk's InfraWorks sits at a larger scale, modelling transport networks and urban context so that planners can evaluate mobility and circulation across a site or district (Bentley Systems, n.d.; Autodesk, n.d.).

Most of these tools rely on some form of agent based simulation, often built on social force ideas, where each simulated pedestrian is pushed and pulled by goals, obstacles and other agents. The behaviour that emerges can be calibrated against observed counts and tuned until the flow looks realistic. This works well for the questions the tools are built for, such as how a concourse clears in an evacuation, or whether a stadium exit meets a capacity standard.

These platforms are powerful, and Motion Pixels is not trying to replace them. The distinction is in what feeds them. Agent based simulation generates movement from assumed rules. It answers what would happen if pedestrians behaved according to the model, and its confidence comes from the plausibility of those rules rather than from a record of a specific afternoon in a specific plaza. It does not begin from how people actually moved through one real space, and it is not designed to. That gap, between simulated behaviour and observed behaviour, is part of what this project is trying to occupy. Motion Pixels does not ask what a crowd would do under a rule set. It asks what a real crowd did, and whether that record predicts itself.

[FIGURE 2.2 HERE]
Source file: THESIS/figures/booklet/bentley_openpaths_practicalreview.png | THESIS/figures/booklet/infraworks_practicalreview.png
Proposed caption: Two commercial approaches to the same problem. Bentley's OpenPaths and LEGION (left) model crowd movement through agent based simulation; Autodesk InfraWorks (right) models mobility across an urban network. Both generate movement from assumed rules rather than from a record of observed behaviour. Screenshots courtesy of Bentley Systems and Autodesk.

## The gaps and positioning

Read together, these three bodies of work leave a clear opening.

Space Syntax explains movement. It provides a configurational account of why some parts of a layout carry more life than others, and it does so from the plan, which is exactly where an architect works. Its reasoning runs from the drawing outward, predicting where movement should concentrate before anyone has walked through the space. What it does not provide is a direct record of what people actually did once the space was built and occupied. It tells you where a street ought to be busy, not who moved along it, at what pace, or where they hesitated. The account is real and measurable, but it stays on the side of potential rather than observed behaviour.

The prediction literature measures movement. It provides models that can continue a trajectory and account for the people around it, and it evaluates them with care against shared benchmarks. What it tends to lack is any grip on a specific architectural setting. The models are trained on a handful of generic scenes, the space enters as background if it enters at all, and the output is judged by an error figure rather than by what it reveals about a place. A model can be excellent at continuing motion in the abstract and still say nothing about the plaza in front of a particular building.

The simulation software simulates movement. It provides planning, crowd modelling and decision support at the scale of a real project, and it is trusted for questions of capacity and safety. What it lacks is a foundation in observed behaviour. Its pedestrians move according to assumed rules, calibrated until the flow looks plausible, so the answer it gives is always conditional on those rules being right. It can tell you what a crowd would do under a model of behaviour, but not what a real crowd did on a real afternoon in the space you are studying.

These three are rarely brought into the same room. Space Syntax and the prediction literature come from different disciplines and cite each other seldom. The simulation tools are commercial and largely separate from both. Each is strong on its own axis and quiet on the others, and the result is that observation, configuration and prediction tend to live apart.

Motion Pixels sits in the space these three leave between them. It starts from observed movement in a real, calibrated site, describes that movement in both behavioural and spatial terms, and then tests whether the description carries enough structure to predict. Predicting movement, in the sense used here, means taking a short window of a real person's motion in a real place and continuing it, then checking that continuation against what the person actually did. The aim is not to win a benchmark or to out simulate a crowd engine. It is to turn a specific space into evidence, and then to ask what that evidence can anticipate, and for how far.

[FIGURE 2.3 HERE]
Source file: THESIS/figures/booklet/gaps.png
Proposed caption: Space Syntax explains, the prediction literature measures, simulation software simulates. Motion Pixels occupies the predictive gap between them, grounded in observed movement.

## Computational pipeline

The rest of the thesis depends on one practical thing. Ordinary video has to become measured movement on an architectural plan. The pipeline that does this is a sequence of steps, each of which was chosen because it earns its place in the argument, not because it is technically interesting on its own.

[FIGURE 2.4 HERE]
Source file: THESIS/figures/booklet/computational_pipeline.png
Proposed caption: The full pipeline, from video and plan through detection, tracking, homography and spatial encoding to the LSTM, its horizon predictions, and the plots and behavioural maps that make the result legible to a designer.

It begins with detection and tracking. Each frame of a recording is passed through an object detector, YOLOv8, which finds the people in it. A tracker, ByteTrack, then links those detections across frames so that each person keeps a stable identity for as long as they stay in view. Getting this to work on real footage needed tuning. The detector expects upright pedestrians, and several recordings were filmed sideways, so each frame is rotated upright before detection and run at high resolution with a low confidence threshold and a high recall tracking setting. The quality of everything downstream depends on how completely the tracker sees the crowd.

Detection gives movement in image pixels. Architecture needs it in metres, on the plan. The second step is calibration by homography. For each recording, points that can be identified both in the footage and on the architectural plan are matched by hand, and from those correspondences a transformation is computed that maps any point in the image to a position on the plan. After calibration, a trajectory that was a line of pixels becomes a path in real coordinates, and distances and speeds become meaningful. Calibration is one of the places the pipeline still depends on manual work, and I return to that limitation in the final chapter.

With movement placed in space, each trajectory is described in two registers at once. The first is behavioural. From the sequence of positions the pipeline derives speed, direction, the distances covered, local density, stops and dwell time. These are the quantities an architect already reasons with when they talk about how a space is used, and they feed the behavioural maps described in the next chapter. The second register is spatial. Using a manually prepared mask of each site, the encoding records where a person is relative to the walkable area, how close they are to obstacles and boundaries, and their affinity to entrances. This is the part that ties a moving body to the architecture around it.

One detail about the metrics matters later. Speed appears in two different unit systems in this project, and they are not interchangeable. Inside the prediction model, motion is measured per step, as the displacement from one frame to the next, because that is what the network predicts. In the behavioural maps, speed is expressed in metres per second, because that is what reads naturally to a person looking at a plan. Both are correct in their place, and I keep them separate so that a number from one is never quietly read as a number from the other.

The result of the encoding is the dataset. Each moving person becomes a sequence of steps, and each step carries both its motion and its spatial situation. That dataset is what the prediction model learns from. In this project the model is a recurrent network, an LSTM, chosen after a comparison described in the next chapter. It reads a short window of a person's recent movement and predicts their next displacement, and by repeating that step it rolls a trajectory forward.

The model is asked to look different distances into the future, and these are the horizons. Through the thesis they are written as H20, H60, H100, H200 and H400, corresponding to roughly 1, 3, 5, 10 and 20 metres of travel. The numbers are frames, not minutes. Testing several horizons rather than one is deliberate, because a method can be reliable at the scale of a step and useless at the scale of a plaza crossing, and the only way to know where that line falls is to look at each range separately.

From the model and the encoded data come the outputs that make the work legible to a designer. Trajectory plots in two and three dimensions, charts, the dataset itself, and the behavioural maps that turn thousands of paths into a readable image of how a space is used. Each of these is a way of handing the same underlying data to a different kind of reading, from the close inspection of a single predicted path to the overview of a whole plaza at once. The pipeline is long, but its logic is single. A video goes in, and a spatial, behavioural, and predictive description of a place comes out.

## The ethical question

None of this is possible without recording people in public space, and that raises a real question rather than a rhetorical one. Is it acceptable to capture and store footage of pedestrians for academic research?

The relevant frame is the General Data Protection Regulation. Article 89(1) allows personal data to be processed for scientific research purposes when appropriate safeguards are in place to protect the rights of individuals, including measures such as data minimisation (European Union, 2016). The design of Motion Pixels fits the spirit of that provision. The system does not identify anyone. It extracts anonymous trajectories and behavioural patterns, and it works at the level of movement across a space rather than the level of a recognisable person. Faces, names and identities are not part of the analysis and are not what the research is about.

Part of the argument is technical rather than legal. The pipeline is built so that identity is discarded early. Detection and tracking assign each person a temporary number that lasts only as long as they are in frame, and the data that survives downstream is a set of coordinates and derived quantities, not images of faces. This is data minimisation in practice. What the research keeps is the movement, not the person who made it, and the movement is what the architectural question is about.

This is easy to overstate, and the source material for this project puts the point more categorically than I would. Article 89 is a safeguards and derogations provision. It sets conditions under which research processing is permitted, but it does not by itself grant blanket permission to record identifiable people. Lawful collection also depends on the controller, the legal basis, and the concrete arrangements around retention, access and anonymisation. Those operational matters, the controller, the legal basis, and the handling of retention, access and anonymisation, were addressed for the recordings used in this work, in keeping with the research safeguards the regulation asks for.

<!-- TRANSITION -->

This chapter has placed Motion Pixels among the work it draws on and set out the machinery it uses. The research reads movement rather than assuming it, returns it to the plan in real measurements, and treats a specific space as the object of study rather than the background. The pipeline that does this, from detection through to the encoded dataset, has been described in principle. What it has not yet done is meet real footage. A method can be coherent on paper and fall apart on the first crowded frame. The next chapter puts the tools to work, first on a single site chosen as a testing ground and then across a set of recordings in Barcelona, and it is there that the argument stops being a proposal and starts producing results.

<!-- /TRANSITION -->

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

# Conclusion

This thesis began with a gap that every architect knows and few can measure. A drawing describes how a space is meant to be used. The people who use it answer with their own movement, and that answer is usually lost once the building is occupied. Motion Pixels was an attempt to hold onto it, to turn ordinary video of a public space into movement that can be measured, placed back onto the plan, and read as evidence.

The path there was not straight. It started on one plaza in front of MACBA, where the first models failed in a way that turned out to be useful. They flattened the curved, turning movement that made the site worth filming, and chasing that failure is what shaped the rest of the work. A capacity test showed the architecture could represent angular movement once it had enough examples, which redirected the research from fixing the model to feeding it. The dataset grew across Barcelona, and the prediction problem was split into horizons so that the method could be judged at each range rather than as a single figure.

What that judgement returned is a bounded claim, and I have tried to keep it bounded throughout. Short-range prediction works. At the scale of a step or two, movement follows reliably from recent motion, and spatial context adds to that once there is enough data to learn from. As the horizon grows the prediction weakens, straightening and falling short, until beyond about five metres it is better read as a direction than a route. Later experiments that changed the training objective recovered part of the missing shape and confirmed that the limits were as much about how the model learned as about how much it saw. None of the results transfer to a genuinely unseen space, and the thesis does not pretend they do.

Set against the question the work opened with, this is enough. The question was whether pedestrian movement carries enough structure to be treated as spatial evidence, and the answer is a qualified yes. It carries that structure most clearly at the scale where an architect actually reasons, at a threshold, an edge, a crossing. Prediction was the test of that structure, not the product. The product, if there is one, is the shift in what an architect can hold. Movement stops being anecdote and becomes a layer that sits next to the plan, described in terms a designer already uses.

Much is still manual, still small, still tied to five spaces in one city. The calibration is drawn by hand, the dataset is modest, and the long-range predictions are weak. These are real limits, and I have kept them visible rather than let a convincing map or a clean interface imply more than the work has earned. A partial tool that is clear about its edges is more useful to a designer than a confident one that is not.

I could keep going on the prediction itself, on ways to shave the error down or push the accuracy toward the centimetre. But the larger actor in this thesis was never the model. It is spatial design. Motion Pixels is an attempt to read the feedback between behaviour and architecture, the quiet back and forth in which a space shapes how people move and their movement, in turn, reveals what the space is actually doing. Behaviour is not a by-product of architecture to be tidied away once a building opens. It is information, and it can feed back into design.

If the work that follows this thesis widens the dataset, sharpens the prediction, and lifts the reading into three dimensions, the point of it will not be a better forecast. It will be a better understanding of the spaces we design, and a slow move toward something worth calling spatial intelligence.

# Bibliography

References follow the Harvard author-date style. Web resources were last checked in September 2026. Where a legal or product source was accessed through a secondary copy, that is noted in the entry; the primary source remains the reference.

Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., and Savarese, S. (2016). [Social LSTM: Human Trajectory Prediction in Crowded Spaces](https://openaccess.thecvf.com/content_cvpr_2016/html/Alahi_Social_LSTM_Human_CVPR_2016_paper.html). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Autodesk (n.d.). [InfraWorks: Mobility and Traffic Simulation documentation](https://help.autodesk.com/view/INFMDR/ENU/). Autodesk product documentation.

Bai, S., Kolter, J. Z., and Koltun, V. (2018). [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). arXiv:1803.01271.

Bentley Systems (n.d.). [LEGION Simulator and OpenPaths](https://www.bentley.com/software/legion-simulator/). Official product description.

European Data Protection Supervisor (2020). [A Preliminary Opinion on Data Protection and Scientific Research](https://www.edps.europa.eu/sites/default/files/publication/20-01-06_opinion_research_en.pdf).

European Union (2016). [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng), especially Articles 5, 6 and 89 and Recital 26. Official Journal of the European Union, L119. Article 89 was consulted through a [public reproduction](https://gdpr-info.eu/art-89-gdpr/); the regulation remains the primary legal reference and project compliance has not been certified.

Giuliari, F., Hasan, I., Cristani, M., and Galasso, F. (2020). [Transformer Networks for Trajectory Forecasting](https://arxiv.org/abs/2003.08111). International Conference on Pattern Recognition (ICPR); arXiv:2003.08111.

Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., and Alahi, A. (2018). [Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks](https://arxiv.org/abs/1803.10892). Proceedings of CVPR.

Hartley, R., and Zisserman, A. (2004). Multiple View Geometry in Computer Vision. 2nd edn. Cambridge: Cambridge University Press. Reference for planar homography estimation used in calibration.

Hillier, B. (1996). [Space Is the Machine: A Configurational Theory of Architecture](https://discovery.ucl.ac.uk/3881/1/SITM.pdf). Cambridge: Cambridge University Press; open edition in UCL Discovery.

Hillier, B., and Hanson, J. (1984). The Social Logic of Space. Cambridge: Cambridge University Press.

Hillier, B., Penn, A., Hanson, J., Grajewski, T., and Xu, J. (1993). [Natural movement: or, configuration and attraction in urban pedestrian movement](https://discovery.ucl.ac.uk/id/eprint/1398/). Environment and Planning B: Planning and Design, 20, 29-66.

Lerner, A., Chrysanthou, Y., and Lischinski, D. (2007). Crowds by Example. Computer Graphics Forum, 26(3), 655-664. Source of the UCY pedestrian benchmark.

MACBA (n.d.). [Architecture and Spaces](https://www.macba.cat/en/architecture-and-spaces/). Museu d'Art Contemporani de Barcelona.

Milieu Consulting (2021, published 2022). [Study on the appropriate safeguards under Article 89(1) GDPR for the processing of personal data for scientific research](https://www.edpb.europa.eu/system/files/2022-01/legalstudy_on_the_appropriate_safeguards_89.1.pdf). Legal study hosted by the European Data Protection Board; not binding law.

Pellegrini, S., Ess, A., Schindler, K., and van Gool, L. (2009). You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking. Proceedings of the International Conference on Computer Vision (ICCV). Source of the ETH pedestrian benchmark.

Project for Public Spaces (n.d.). [A Primer on Seating](https://www.pps.org/article/generalseating). Interpretation of Whyte's work used for the seating discussion.

Robicquet, A., Sadeghian, A., Alahi, A., and Savarese, S. (2016). Learning Social Etiquette: Human Trajectory Understanding in Crowded Scenes. Proceedings of the European Conference on Computer Vision (ECCV). Source of the Stanford Drone Dataset.

Sadeghian, A., Kosaraju, V., Sadeghian, A., Hirose, N., Rezatofighi, H., and Savarese, S. (2019). [SoPhie: An Attentive GAN for Predicting Paths Compliant to Social and Physical Constraints](https://arxiv.org/abs/1806.01482). Proceedings of CVPR.

Ultralytics (n.d.). [YOLOv8 documentation](https://docs.ultralytics.com/models/yolov8/). Official model documentation for the detector used in the pipeline.

Whyte, W. H. (1980). The Social Life of Small Urban Spaces. Washington, DC: The Conservation Foundation. Discussion in this book draws on the inspected [Project for Public Spaces account](https://www.pps.org/product/the-social-life-of-small-urban-spaces); no page-specific quotation is used.

Yuan, Y., Weng, X., Ou, Y., and Kitani, K. (2021). [AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting](https://arxiv.org/abs/2103.14023). Proceedings of ICCV.

Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., and Wang, X. (2022). [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864). Proceedings of ECCV. The tracker used in the pipeline.
