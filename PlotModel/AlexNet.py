import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input("Marc_Chagall_180.jpg"),
    to_Conv("conv_input", 224, 3, offset="(-1,0,0)", to="(0,0,0)", height=22.4, depth=22.4, width=2),
    
    # Block 1
    to_Conv("conv_b1", 56, 96, offset="(1,0,0)", to="(conv_input-east)", height=5.6, depth=5.6, width=9.6),
    to_connection("conv_input", "conv_b1"),
    to_Conv("BatchNorm_b1", 56, 96, offset="(0,0,0)", to="(conv_b1-east)", caption="Batch\\\\Norm", height=5.6, depth=5.6, width=9.6),
    to_Pool("pool_b1", offset="(0,0,0)", to="(BatchNorm_b1-east)", height=2.7, depth=2.7, width=12.8),
    
    # Block 2
    to_Conv("conv_b2", 27, 256, offset="(1,0,0)", to="(pool_b1-east)", height=2.7, depth=2.7, width=12.8),
    to_connection("pool_b1", "conv_b2"),
    to_Conv("BatchNorm_b2", 27, 64, offset="(0,0,0)", to="(conv_b2-east)", caption="Batch\\\\Norm", height=2.7, depth=2.7, width=3.2),
    to_Pool("pool_b2", offset="(0,0,0)", to="(BatchNorm_b2-east)", height=1.3, depth=1.3, width=6.4),
    
    # Block 3
    to_Conv("conv1_b3", 13, 384, offset="(1,0,0)", to="(pool_b2-east)", height=1.3, depth=1.3, width=19.2),
    to_connection("pool_b2", "conv1_b3"),
    to_Conv("BatchNorm1_b3", 13, 384, offset="(0,0,0)", to="(conv1_b3-east)", caption="Batch\\\\Norm", height=1.3, depth=1.3, width=19.2),
    to_Conv("conv2_b3", 13, 384, offset="(0,0,0)", to="(BatchNorm1_b3-east)", height=1.3, depth=1.3, width=19.2),
    to_Conv("BatchNorm2_b3", 13, 384, offset="(0,0,0)", to="(conv2_b3-east)", caption="Batch\\\\Norm", height=1.3, depth=1.3, width=19.2),
    to_Conv("conv3_b3", 13, 256, offset="(0,0,0)", to="(BatchNorm2_b3-east)", height=1.3, depth=1.3, width=12.8),
    to_Conv("BatchNorm3_b3", 6, 256, offset="(0,0,0)", to="(conv3_b3-east)", caption="Batch\\\\Norm", height=1.3, depth=1.3, width=12.8),
    to_Pool("pool_b3", offset="(0,0,0)", to="(BatchNorm3_b3-east)", height=0.6, depth=0.6, width=12.8),
    
    to_SoftMax("Flatten", 9216 , "(1,0,0)", "(pool_b3-east)", caption="Flatten", depth=40),
    to_connection("pool_b3", "Flatten"),
    to_SoftMax("Dense", 4096 , "(1,0,0)", "(Flatten-east)", caption="Dense", depth=20.48),
    to_connection("Flatten", "Dense"),
    to_SoftMax("Dropout", 4096 , "(1,0,0)", "(Dense-east)", caption="Dropout", depth=20.48),
    to_connection("Dense", "Dropout"),
    to_SoftMax("Dense", 4096 , "(1,0,0)", "(Flatten-east)", caption="Dense", depth=20.48),
    to_connection("Flatten", "Dense"),
    to_SoftMax("Dropout", 4096 , "(1,0,0)", "(Dense-east)", caption="Dropout", depth=20.48),
    to_connection("Dense", "Dropout"),
    to_SoftMax("SoftMax", 8 , "(1,0,0)", "(Dropout-east)", caption="SoftMax", depth=8),
    to_connection("Dropout", "SoftMax"),    
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
