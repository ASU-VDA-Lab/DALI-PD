# synthetic_bnechmark
![Inference](../etc/example.png)
This directory contains 23,069 synthetic circuit heatmaps for the following categories:
 - [cell_density](./cell_density/): The cell density map is a distribution of the number of cells in a unit area, as shown in Fig.(a).
 - [macro_region](./macro_region/): The macro region map is a binary map that shows the regions on the chip that are occupied by macros, as shown in Fig.(b).
 - [RUDY](./RUDY/): RUDY heatmaps are used for early routing demand estimation after placement and widely adopted to estimate routing congestion. The RUDY heatmap is shown in Fig.(c).
 - [IR_drop](./IR_drop/): IR drop is the voltage drop across the chip through the power grid network. A high IR drop indicates a high potential for logic errors and timing violations. A sample heatmap for the IR drop distribution is shown in Fig.~\ref{fig:features}(d).
 - [power_all](./power_all/): The power heatmaps simply sum the leakage power, switching power and internal power, as shown in Fig.(e).
 - [power_sca](./power_sca/): High power values indicate high current demand, which correlates with potential IR drop when switching and internal power are scaled by toggle rate, as shown in Fig.(f).

## Decompress
```
bash decompress.sh ./cell_density ./cell_density_restored
bash decompress.sh ./macro_region ./macro_region_restored
bash decompress.sh ./RUDY ./RUDY_restored
bash decompress.sh ./IR_drop ./IR_drop_restored
bash decompress.sh ./power_all ./power_all_restored
bash decompress.sh ./power_sca ./power_sca_restored
```