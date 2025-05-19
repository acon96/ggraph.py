
dot -Kdot -Tpng -oqwen3.png graph.dot -Gnslimit=20 -Gmaxiter=1000 -Gstart=random -Gnojustify=true -Goutputorder=edgesfirst -Gsplines=false -Grankdir=TB -v3
dot -Kdot -Tpng -oqwen3_llamacpp.png qwen3_llamacpp.dot -Gnslimit=20 -Gmaxiter=1000 -Gstart=random -Gnojustify=true -Goutputorder=edgesfirst -Gsplines=false -v3