import pandas as pd
import streamlit as st
import pydeck
from typing import List, Dict

from data.tsp_data import get_tsp_data
from algos.tsp import TSPPopulation, TSPCandidate


N_ITERATIONS = 2000
CITIES = get_tsp_data()


def get_candidate_route_coordinates(candidate: TSPCandidate) -> List[Dict[str, float]]:

    cities_ordered = CITIES.iloc[candidate.chromosomes]

    route = [
        *[
            {
                "start": (cities_ordered.iloc[i]["longitude"], cities_ordered.iloc[i]["latitude"]),
                "end": (cities_ordered.iloc[i + 1]["longitude"], cities_ordered.iloc[i + 1]["latitude"]),
            } for i in range(len(cities_ordered) - 1)
        ],
        {
            "start": (cities_ordered.iloc[-1]["longitude"], cities_ordered.iloc[-1]["latitude"]),
            "end": (cities_ordered.iloc[0]["longitude"], cities_ordered.iloc[0]["latitude"]),
        }
     ]

    return route


def main():

    st.title("Genetic algorithm for the Traveling Salesman Problem")
    launch_button = st.button("Launch algo")
    progress_bar = st.progress(0)
    map = st.empty()
    chart = st.empty()

    map.pydeck_chart(pydeck.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pydeck.ViewState(
            latitude=CITIES["latitude"].mean(),
            longitude=CITIES["longitude"].mean(),
            zoom=2.7,
        ),
        layers=[
            pydeck.Layer(
                "ScatterplotLayer",
                data=CITIES,
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                get_radius=50000,
                get_fill_color=[180, 0, 200, 140],
                pickable=True
            ),
        ],
    ))

    tracking_statistics = []

    population = TSPPopulation()

    best_solution = min(population.candidates, key=lambda c: c.fitness_score)
    best_score = best_solution.fitness_score
    tracking_statistics.append(population.statistics)

    map.pydeck_chart(pydeck.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pydeck.ViewState(
            latitude=CITIES["latitude"].mean(),
            longitude=CITIES["longitude"].mean(),
            zoom=2.7,
        ),
        layers=[
            pydeck.Layer(
                "ScatterplotLayer",
                data=CITIES,
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                get_radius=50000,
                get_fill_color=[180, 0, 200, 140],
                pickable=True
            ),
            pydeck.Layer(
                "LineLayer",
                get_candidate_route_coordinates(best_solution),
                get_source_position="start",
                get_target_position="end",
                get_color=[255, 0, 0],
                coverage=1,
                width_scale=3,
            )
        ],
    ))

    if launch_button:

        for i in range(N_ITERATIONS):
            progress_bar.progress((i + 1) / N_ITERATIONS)
            population.evolve()

            tracking_statistics.append(population.statistics)

            if population.candidates[0].fitness_score < best_score:
                best_solution = population.candidates[0]
                best_score = best_solution.fitness_score

            map.pydeck_chart(pydeck.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pydeck.ViewState(
                    latitude=CITIES["latitude"].mean(),
                    longitude=CITIES["longitude"].mean(),
                    zoom=2.7,
                ),
                layers=[
                    pydeck.Layer(
                        "ScatterplotLayer",
                        data=CITIES,
                        get_position=["longitude", "latitude"],
                        auto_highlight=True,
                        get_radius=50000,
                        get_fill_color=[180, 0, 200, 140],
                        pickable=True
                    ),
                    pydeck.Layer(
                        "LineLayer",
                        get_candidate_route_coordinates(
                            population.candidates[len(population.candidates) // 2]
                        ),
                        get_source_position="start",
                        get_target_position="end",
                        get_color=[169, 169, 169],
                        coverage=1,
                        width_scale=1,
                    ),
                    pydeck.Layer(
                        "LineLayer",
                        get_candidate_route_coordinates(best_solution),
                        get_source_position="start",
                        get_target_position="end",
                        get_color=[255, 0, 0],
                        coverage=1,
                        width_scale=3,
                    ),
                ],
            ))

            tracking_statistics_df = pd.DataFrame(
                tracking_statistics,
                columns=["min", "mean", "median", "max"]
            )

            chart.line_chart(tracking_statistics_df)

        map.pydeck_chart(pydeck.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pydeck.ViewState(
                latitude=CITIES["latitude"].mean(),
                longitude=CITIES["longitude"].mean(),
                zoom=2.7,
            ),
            layers=[
                pydeck.Layer(
                    "ScatterplotLayer",
                    data=CITIES,
                    get_position=["longitude", "latitude"],
                    auto_highlight=True,
                    get_radius=50000,
                    get_fill_color=[180, 0, 200, 140],
                    pickable=True
                ),
                pydeck.Layer(
                    "LineLayer",
                    get_candidate_route_coordinates(best_solution),
                    get_source_position="start",
                    get_target_position="end",
                    get_color=[255, 0, 0],
                    coverage=1,
                    width_scale=3,
                ),
            ],
        ))


if __name__ == "__main__":
    main()
