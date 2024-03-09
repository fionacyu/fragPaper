import sys

import data.input
import charac.mgraph
import charac.boxing
import charac.gcharac
import opt.optimise

def main():
    input_name: str = sys.argv[1]

    if (len(sys.argv) != 2):
        raise SystemExit("Usage: ./main.py <input>")
    
    boxarray: charac.boxing.mbox_array = charac.boxing.mbox_array()
    graph: charac.mgraph.mgraph = charac.mgraph.mgraph()

    data.input.read_input(graph, input_name)
    boxarray.define_boxes(graph)
    charac.gcharac.characterise_graph(graph, boxarray)
    optimiser: opt.optimise.Optimiser_Frag = opt.optimise.Optimiser_Frag(graph, boxarray)
    optimiser.run()


if __name__ == "__main__":
    main()