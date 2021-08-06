import pandas as pd
import streamlit as st
import pydeck

from data.tsp_data import get_tsp_data


def main():

    tsp_data = get_tsp_data()
    distance = []

    st.title("Genetic algorithm for the Traveling Salesman Problem")
    launch_button = st.button("Launch algo")
    map = st.empty()
    chart = st.empty()

    map.pydeck_chart(pydeck.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pydeck.ViewState(
            latitude=tsp_data["latitude"].mean(),
            longitude=tsp_data["longitude"].mean(),
            zoom=2.7,
        ),
        layers=[
            pydeck.Layer(
                "ScatterplotLayer",
                data=tsp_data,
                get_position=["longitude", "latitude"],
                auto_highlight=True,
                get_radius=50000,
                get_fill_color=[180, 0, 200, 140],
                pickable=True
            ),
        ],
    ))

    chart.line_chart(distance)

    if launch_button:

        for i in range(10):
            import time
            time.sleep(1)

            tsp_data = tsp_data.sample(frac=1)

            test = [
                {
                    "start": (tsp_data.iloc[i]["longitude"], tsp_data.iloc[i]["latitude"]),
                    "end": (tsp_data.iloc[i+1]["longitude"], tsp_data.iloc[i+1]["latitude"]),
                } for i in range(len(tsp_data) - 1)
            ]
            test.append(
                {
                    "start": (tsp_data.iloc[-1]["longitude"], tsp_data.iloc[-1]["latitude"]),
                    "end": (tsp_data.iloc[0]["longitude"], tsp_data.iloc[0]["latitude"]),
                }
            )

            distance.append((i ** 2, - i ** 3))
            df = pd.DataFrame(distance, columns=["min", "max"])

            map.pydeck_chart(pydeck.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pydeck.ViewState(
                    latitude=tsp_data["latitude"].mean(),
                    longitude=tsp_data["longitude"].mean(),
                    zoom=2.7,
                ),
                layers=[
                    pydeck.Layer(
                        "ScatterplotLayer",
                        data=tsp_data,
                        get_position=["longitude", "latitude"],
                        auto_highlight=True,
                        get_radius=50000,
                        get_fill_color=[180, 0, 200, 140],
                        pickable=True
                    ),
                    pydeck.Layer(
                        "LineLayer",
                        test,
                        get_source_position="start",
                        get_target_position="end",
                        get_color=[255, 0, 0],
                        coverage=1,
                        width_scale=3,
                    )
                ],
            ))

            chart.line_chart(df)

# TODO : work-in-progress
"""
def solve_tsp():

    tracking_statistics = []

    print(f"Iteration 0 / {N_ITERATIONS}")

    population = TSPPopulation()
    print(population.statistics)

    best_solution = min(population.candidates, key=lambda c: c.fitness_score)
    best_score = best_solution.fitness_score
    tracking_statistics.append(population.statistics)

    for i in range(N_ITERATIONS):
        print(f"Iteration {i + 1} / {N_ITERATIONS}")
        population.evolve()

        print(population.statistics)
        tracking_statistics.append(population.statistics)

        if population.candidates[0].fitness_score < best_score:
            best_solution = population.candidates[0]
            best_score = best_solution.fitness_score

    tracking_statistics = np.array(tracking_statistics)
    x_list = np.arange(len(tracking_statistics))
    plt.plot(x_list, tracking_statistics[:, 0], label="min")
    plt.plot(x_list, tracking_statistics[:, 1], label="mean")
    plt.plot(x_list, tracking_statistics[:, 2], label="median")
    plt.plot(x_list, tracking_statistics[:, 3], label="max")

    plt.legend()
    plt.show()

    plot_candidate(best_solution)
"""


if __name__ == "__main__":
    main()
