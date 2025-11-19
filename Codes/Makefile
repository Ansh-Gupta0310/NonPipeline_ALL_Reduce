CC = mpicc
CFLAGS = -Wall -O3 -lm
TARGETS = linear_exchange ring rabenseifner non_pipeline_allreduce

.PHONY: all clean

all: $(TARGETS)

linear_exchange: linear_exchange.c
	$(CC) $(CFLAGS) -o $@ $<

ring: ring.c
	$(CC) $(CFLAGS) -o $@ $<

rabenseifner: rabenseifner.c
	$(CC) $(CFLAGS) -o $@ $<

non_pipeline_allreduce: non_pipeline_allreduce.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o *.csv latency_results.csv
	rm -rf latency_graphs latency_comparison_graphs.zip