import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),
    to_input("Titian_170.jpg"),
    to_Conv("conv_input", 224, 3, offset="(-1,0,0)", to="(0,0,0)", height=22.4, depth=22.4, width=2),
    
    # Block 1
    to_Conv("conv_b1", 224, 32, offset="(0,0,0)", to="(conv_input-east)", height=22.4, depth=22.4, width=6),
    to_Conv("InstanceNorm_b1", 224, 32, offset="(0,0,0)", to="(conv_b1-east)", caption="Instance\\\\Norm", height=22.4, depth=22.4, width=6),
    to_Pool("pool_b1", offset="(0,0,0)", to="(InstanceNorm_b1-east)", height=11.2, depth=11.2, width=6),
    
    # Block 2
    to_Conv("conv_b2", 112, 64, offset="(0,0,0)", to="(pool_b1-east)", height=11.2, depth=11.2, width=8),
    to_Conv("InstanceNorm_b2", 112, 64, offset="(0,0,0)", to="(conv_b2-east)", caption="Instance\\\\Norm", height=11.2, depth=11.2, width=8),
    to_Pool("pool_b2", offset="(0,0,0)", to="(InstanceNorm_b2-east)", height=5.6, depth=5.6, width=8),
    
    # Block 3
    to_Conv("conv_b3", 56, 128, offset="(0,0,0)", to="(pool_b2-east)", height=5.6, depth=5.6, width=12),
    to_Conv("InstanceNorm_b3", 56, 128, offset="(0,0,0)", to="(conv_b3-east)", caption="Instance\\\\Norm", height=5.6, depth=5.6, width=12),
    to_Pool("pool_b3", offset="(0,0,0)", to="(InstanceNorm_b3-east)", height=2.8, depth=2.8, width=12),
    
    # Block 4
    to_Conv("conv_b4", 28, 64, offset="(0,0,0)", to="(pool_b3-east)", height=2.8, depth=2.8, width=8),
    to_Conv("InstanceNorm_b4", 28, 64, offset="(0,0,0)", to="(conv_b4-east)", caption="Instance\\\\Norm", height=2.8, depth=2.8, width=8),
    to_Pool("pool_b4", offset="(0,0,0)", to="(InstanceNorm_b4-east)", height=1.4, depth=1.4, width=7),
    
    to_SoftMax("Flatten", 12544 , "(1,0,0)", "(pool_b4-east)", caption="Flatten", depth=40),
    to_connection("pool_b4", "Flatten"),
    to_SoftMax("Dense", 64 , "(1,0,0)", "(Flatten-east)", caption="Dense", depth=16),
    to_connection("Flatten", "Dense"),
    to_SoftMax("BatchNorm_b5", 64 , "(1,0,0)", "(Dense-east)", caption="Batch\\\\Norm", depth=16),
    to_connection("Dense", "BatchNorm_b5"),
    to_SoftMax("Dropout", 64 , "(1,0,0)", "(BatchNorm_b5-east)", caption="Dropout (0.2)", depth=16),
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
