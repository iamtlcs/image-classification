import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input("Rene_Magritte_87.jpg"),
    to_Conv("conv_input", 224, 3, offset="(-1,0,0)", to="(0,0,0)", height=22.4, depth=22.4, width=1),
    
    # Block 1
    to_Conv("conv_b1", 224, 32, offset="(0,0,0)", to="(conv_input-east)", height=22.4, depth=22.4, width=2),
    to_Pool("pool_b1", offset="(0,0,0)", to="(conv_b1-east)", height=11.2, depth=11.2, width=2),
    
    # Block 2
    to_Conv("conv_b2", 112, 32, offset="(0,0,0)", to="(pool_b1-east)", height=11.2, depth=11.2, width=2),
    to_Pool("pool_b2", offset="(0.5,0,0)", to="(conv_b2-east)", height=5.6, depth=5.6, width=6),
    
    # Block 3
    to_Conv("conv_b3", 56, 32, offset="(0,0,0)", to="(pool_b2-east)", height=5.6, depth=5.6, width=2),
    to_Pool("pool_b3", offset="(0,0,0)", to="(conv_b3-east)", height=2.8, depth=2.8, width=7),
    
    # Block 4
    to_Conv("conv_b4", 28, 32, offset="(0,0,0)", to="(pool_b3-east)", height=2.8, depth=2.8, width=2),
    to_Pool("pool_b4", offset="(0,0,0)", to="(conv_b4-east)", height=1.4, depth=1.4, width=2),
    
    # Block 5
    to_Conv("conv_b4", 14, 32, offset="(0,0,0)", to="(pool_b4-east)", height=1.4, depth=1.4, width=2),
    to_Pool("pool_b4", offset="(0,0,0)", to="(conv_b4-east)", height=0.7, depth=0.7, width=2),
    
    # Block 6
    to_Conv("Reshape", 49, 32, offset="(1.5,0,0)", to="(pool_b4-east)", caption="Reshape", height=4.9, depth=3.2, width=0.1),
    to_connection("pool_b4", "Reshape"),
    to_Conv("LSTM", 49, 32, offset="(1.5,0,0)", to="(Reshape-east)", caption="LSTM", height=4.9, depth=3.2, width=0.1),
    to_connection("Reshape", "LSTM"),
    
    # Block 7
    to_SoftMax("Flatten", 1568 , "(3,0,0)", "(LSTM-east)", caption="Flatten", depth=64),
    to_connection("LSTM", "Flatten"),
    to_SoftMax("SoftMax", 8 , "(1,0,0)", "(Flatten-east)", caption="SoftMax", depth=8),
    to_connection("Flatten", "SoftMax"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
