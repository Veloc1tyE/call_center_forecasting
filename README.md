# call_center_forecasting
A Deep RL Agent that can learn resource allocation for call-centers

This agent learns through exploring its environment, and has very natural error signals
- the proportion of customers who are not services within the desired timeframe
- punishment for over-utilisation of resources
- bonus for 'sweet-spot' optimisation

It implements Policy-Gradient optimisation and shows itself to be very effective at optimising underperforming systems
