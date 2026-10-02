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

---

*Target word count: 2,000. Actual word count: see chapter audit in 01_SOURCE_MAP.md.*
