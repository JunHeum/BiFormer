import cupy
import torch
import re
import math

correlation_forward = '''
    extern "C" __global__ void correlation_forward(
        const int n,
        const float* feature1,
        const float* feature2,
        const float* flow,
        const float* time,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        float fltOutput = 0.0;
    
        const int intN  = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY  = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX  = ( intIndex                                                    ) % SIZE_3(output);


        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;
        
        float t = VALUE_2(time,intN, 0);
        
        float ratio_x = (float) (SIZE_3(feature1)) / SIZE_3(flow);
        float ratio_y = (float) (SIZE_2(feature1)) / SIZE_2(flow);

        float fltX1 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) * -2 *       t) * ratio_x - k * -2 *      t;
        float fltY1 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) * -2 *       t) * ratio_y - l * -2 *      t;
        float fltX2 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) *  2 * (1 - t)) * ratio_x - k *  2 * (1 - t);
        float fltY2 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) *  2 * (1 - t)) * ratio_y - l *  2 * (1 - t);
                
        int intLX1 = (int) (floor(fltX1));
        int intTY1 = (int) (floor(fltY1));
        int intRX1 = intLX1 + 1;
        int intBY1 = intTY1 + 1;
        
        int intRX2 = (int) (ceil(fltX2));
        int intBY2 = (int) (ceil(fltY2));
        int intLX2 = intRX2 - 1;
        int intTY2 = intBY2 - 1;
        
        float fltnw = ((float) intRX1 - fltX1) * ((float) intBY1 - fltY1);
        float fltne = (fltX1 - (float) intLX1) * ((float) intBY1 - fltY1);
        float fltsw = ((float) intRX1 - fltX1) * (fltY1 - (float) intTY1);
        float fltse = (fltX1 - (float) intLX1) * (fltY1 - (float) intTY1);
        
        if ((intRX1 >= 0) & (intBY1 >= 0) & (intLX1 < SIZE_3(feature1)) & (intTY1 < SIZE_2(feature1))) {
            if ((intRX2 >= 0) & (intBY2 >= 0) & (intLX2 < SIZE_3(feature1)) & (intTY2 < SIZE_2(feature1))) {
                for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                    if ((intLX1 >= 0) & (intTY1 >= 0) & (intRX2 < SIZE_3(feature1)) & (intBY2 < SIZE_2(feature1))) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intTY1, intLX1) * VALUE_4(feature2, intN, intChannel, intBY2, intRX2) * fltnw;
                    }
                    if ((intLX1 >= 0) & (intBY1 < SIZE_2(feature1)) & (intRX2 < SIZE_3(feature2)) & (intTY2 >= 0)) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intBY1, intLX1) * VALUE_4(feature2, intN, intChannel, intTY2, intRX2) * fltsw;
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intTY1 >= 0) & (intLX2 >= 0) & (intBY2 < SIZE_2(feature1))) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intTY1, intRX1) * VALUE_4(feature2, intN, intChannel, intBY2, intLX2) * fltne;
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intBY1 < SIZE_2(feature1)) & (intLX2 >= 0) & (intTY2 >= 0)) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intBY1, intRX1) * VALUE_4(feature2, intN, intChannel, intTY2, intLX2) * fltse;
                    }
                }
            }
        }
        output[intIndex] = fltOutput;
    } }
'''

correlation_coords_forward = '''
    extern "C" __global__ void correlation_coords_forward(
        const int n,
        const float* feature1,
        const float* feature2,
        const float* coords_bw,
        const float* coords_fw,
        const float* time,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        float fltOutput = 0.0;
    
        const int intN  = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY  = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX  = ( intIndex                                                    ) % SIZE_3(output);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;
        
        float t = VALUE_2(time,intN, 0);

        float fltX1 = VALUE_4(coords_bw, intN, 0, intY, intX) - (-2 *    t  * k);
                
        int intLX1 = (int) (floor(fltX1));
        int intTY1 = (int) (floor(fltY1));
        int intRX1 = intLX1 + 1;
        int intBY1 = intTY1 + 1;
        
        int intRX2 = (int) (ceil(fltX2));
        int intBY2 = (int) (ceil(fltY2));
        int intLX2 = intRX2 - 1;
        int intTY2 = intBY2 - 1;
        
        float fltnw = ((float) intRX1 - fltX1) * ((float) intBY1 - fltY1);
        float fltne = (fltX1 - (float) intLX1) * ((float) intBY1 - fltY1);
        float fltsw = ((float) intRX1 - fltX1) * (fltY1 - (float) intTY1);
        float fltse = (fltX1 - (float) intLX1) * (fltY1 - (float) intTY1);
        
        if ((intRX1 >= 0) & (intBY1 >= 0) & (intLX1 < SIZE_3(output)) & (intTY1 < SIZE_2(output))) {
            if ((intRX2 >= 0) & (intBY2 >= 0) & (intLX2 < SIZE_3(output)) & (intTY2 < SIZE_2(output))) {
                for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                    if ((intLX1 >= 0) & (intTY1 >= 0) & (intRX2 < SIZE_3(output)) & (intBY2 < SIZE_2(output))) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intTY1, intLX1) * VALUE_4(feature2, intN, intChannel, intBY2, intRX2) * fltnw;
                    }
                    if ((intLX1 >= 0) & (intBY1 < SIZE_2(feature1)) & (intRX2 < SIZE_3(feature2)) & (intTY2 >= 0)) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intBY1, intLX1) * VALUE_4(feature2, intN, intChannel, intTY2, intRX2) * fltsw;
                    }
                    if ((intRX1 < SIZE_3(output)) & (intTY1 >= 0) & (intLX2 >= 0) & (intBY2 < SIZE_2(output))) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intTY1, intRX1) * VALUE_4(feature2, intN, intChannel, intBY2, intLX2) * fltne;
                    }
                    if ((intRX1 < SIZE_3(output)) & (intBY1 < SIZE_2(output)) & (intLX2 >= 0) & (intTY2 >= 0)) {
                        fltOutput += VALUE_4(feature1, intN, intChannel, intBY1, intRX1) * VALUE_4(feature2, intN, intChannel, intTY2, intLX2) * fltse;
                    }
                }
            }
        }
        output[intIndex] = fltOutput;
    } }
'''

correlation_backward_feature = '''
    extern "C" __global__ void correlation_backward_feature(
        const int n,
        const float* gradLoss,
        const float* feature1,
        const float* feature2,
        const float* flow,
        const float* time,
        float* gradInput1,
        float* gradInput2
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        const int intN  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss) / SIZE_1(gradLoss) ) % SIZE_0(gradLoss);
        const int intC  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss)                    ) % SIZE_1(gradLoss);
        const int intY  = ( intIndex / SIZE_3(gradLoss)                                       ) % SIZE_2(gradLoss);
        const int intX  = ( intIndex                                                          ) % SIZE_3(gradLoss);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;

        float t = VALUE_2(time,intN, 0);
        
        float ratio_x = (float) (SIZE_3(feature1)) / SIZE_3(flow);
        float ratio_y = (float) (SIZE_2(feature1)) / SIZE_2(flow);

        float fltX1 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) * -2 *       t) * ratio_x - k * -2 *      t;
        float fltY1 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) * -2 *       t) * ratio_y - l * -2 *      t;
        float fltX2 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) *  2 * (1 - t)) * ratio_x - k *  2 * (1 - t);
        float fltY2 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) *  2 * (1 - t)) * ratio_y - l *  2 * (1 - t);
        
        int intLX1 = (int) (floor(fltX1));
        int intTY1 = (int) (floor(fltY1));
        int intRX1 = intLX1 + 1;
        int intBY1 = intTY1 + 1;
        
        int intRX2 = (int) (ceil(fltX2));
        int intBY2 = (int) (ceil(fltY2));
        int intLX2 = intRX2 - 1;
        int intTY2 = intBY2 - 1;
        
        float fltnw = ((float) intRX1 - fltX1) * ((float) intBY1 - fltY1);
        float fltne = (fltX1 - (float) intLX1) * ((float) intBY1 - fltY1);
        float fltsw = ((float) intRX1 - fltX1) * (fltY1 - (float) intTY1);
        float fltse = (fltX1 - (float) intLX1) * (fltY1 - (float) intTY1);
        
        if ((intRX1 >= 0) & (intBY1 >= 0) & (intLX1 < SIZE_3(feature1)) & (intTY1 < SIZE_2(feature1))) {
            if ((intRX2 >= 0) & (intBY2 >= 0) & (intLX2 < SIZE_3(feature1)) & (intTY2 < SIZE_2(feature1))) {
                float fltLoss = VALUE_4(gradLoss, intN, intC, intY, intX);
                for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                    if ((intLX1 >= 0) & (intTY1 >= 0) & (intRX2 < SIZE_3(feature1)) & (intBY2 < SIZE_2(feature1))) {
                        atomicAdd(&gradInput1[OFFSET_4(gradInput1, intN, intChannel, intTY1, intLX1)], VALUE_4(feature2, intN, intChannel, intBY2, intRX2) * fltnw * fltLoss);
                        atomicAdd(&gradInput2[OFFSET_4(gradInput2, intN, intChannel, intBY2, intRX2)], VALUE_4(feature1, intN, intChannel, intTY1, intLX1) * fltnw * fltLoss);
                    }
                    if ((intLX1 >= 0) & (intBY1 < SIZE_2(feature1)) & (intRX2 < SIZE_3(feature2)) & (intTY2 >= 0)) {
                        atomicAdd(&gradInput1[OFFSET_4(gradInput1, intN, intChannel, intBY1, intLX1)], VALUE_4(feature2, intN, intChannel, intTY2, intRX2) * fltsw * fltLoss);
                        atomicAdd(&gradInput2[OFFSET_4(gradInput2, intN, intChannel, intTY2, intRX2)], VALUE_4(feature1, intN, intChannel, intBY1, intLX1) * fltsw * fltLoss);
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intTY1 >= 0) & (intLX2 >= 0) & (intBY2 < SIZE_2(feature1))) {
                        atomicAdd(&gradInput1[OFFSET_4(gradInput1, intN, intChannel, intTY1, intRX1)], VALUE_4(feature2, intN, intChannel, intBY2, intLX2) * fltne * fltLoss);
                        atomicAdd(&gradInput2[OFFSET_4(gradInput2, intN, intChannel, intBY2, intLX2)], VALUE_4(feature1, intN, intChannel, intTY1, intRX1) * fltne * fltLoss);
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intBY1 < SIZE_2(feature1)) & (intLX2 >= 0) & (intTY2 >= 0)) {
                        atomicAdd(&gradInput1[OFFSET_4(gradInput1, intN, intChannel, intBY1, intRX1)], VALUE_4(feature2, intN, intChannel, intTY2, intLX2) * fltse * fltLoss);
                        atomicAdd(&gradInput2[OFFSET_4(gradInput2, intN, intChannel, intTY2, intLX2)], VALUE_4(feature1, intN, intChannel, intBY1, intRX1) * fltse * fltLoss);
                    }
                }
            }
        }
    } }
'''

correlation_backward_flow = '''
    extern "C" __global__ void correlation_backward_flow(
        const int n,
        const float* gradLoss,
        const float* feature1,
        const float* feature2,
        const float* flow,
        const float* time,
        float* gradFlow
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {

        const int intN  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss) / SIZE_1(gradLoss) ) % SIZE_0(gradLoss);
        const int intC  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss)                    ) % SIZE_1(gradLoss);
        const int intY  = ( intIndex / SIZE_3(gradLoss)                                       ) % SIZE_2(gradLoss);
        const int intX  = ( intIndex                                                          ) % SIZE_3(gradLoss);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;
        
        float t = VALUE_2(time,intN, 0);   
             
        float ratio_x = (float) (SIZE_3(feature1)) / SIZE_3(flow);
        float ratio_y = (float) (SIZE_2(feature1)) / SIZE_2(flow);

        float fltX1 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) * -2 *       t) * ratio_x - k * -2 *      t;
        float fltY1 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) * -2 *       t) * ratio_y - l * -2 *      t;
        float fltX2 = ((float) intX + VALUE_4(flow, intN, 0, intY, intX) *  2 * (1 - t)) * ratio_x - k *  2 * (1 - t);
        float fltY2 = ((float) intY + VALUE_4(flow, intN, 1, intY, intX) *  2 * (1 - t)) * ratio_y - l *  2 * (1 - t);
        
        int intLX1 = (int) (floor(fltX1));
        int intTY1 = (int) (floor(fltY1));
        int intRX1 = intLX1 + 1;
        int intBY1 = intTY1 + 1;
        
        int intRX2 = (int) (ceil(fltX2));
        int intBY2 = (int) (ceil(fltY2));
        int intLX2 = intRX2 - 1;
        int intTY2 = intBY2 - 1;
        
        float fltnw = ((float) intRX1 - fltX1) * ((float) intBY1 - fltY1);
        float fltne = (fltX1 - (float) intLX1) * ((float) intBY1 - fltY1);
        float fltsw = ((float) intRX1 - fltX1) * (fltY1 - (float) intTY1);
        float fltse = (fltX1 - (float) intLX1) * (fltY1 - (float) intTY1);
        
        float fltnwx = (-1.0) * ((float) intBY1 - fltY1) * (-2.0) * t * ratio_x;
        float fltnex = (+1.0) * ((float) intBY1 - fltY1) * (-2.0) * t * ratio_x;
        float fltswx = (-1.0) * (fltY1 - (float) intTY1) * (-2.0) * t * ratio_x;
        float fltsex = (+1.0) * (fltY1 - (float) intTY1) * (-2.0) * t * ratio_x;
        
        float fltnwy = ((float) intRX1 - fltX1) * (-1.0) * (-2.0) * t * ratio_y;
        float fltney = (fltX1 - (float) intLX1) * (-1.0) * (-2.0) * t * ratio_y;
        float fltswy = ((float) intRX1 - fltX1) * (+1.0) * (-2.0) * t * ratio_y;
        float fltsey = (fltX1 - (float) intLX1) * (+1.0) * (-2.0) * t * ratio_y;


        if ((intRX1 >= 0) & (intBY1 >= 0) & (intLX1 < SIZE_3(feature1)) & (intTY1 < SIZE_2(feature1))) {
            if ((intRX2 >= 0) & (intBY2 >= 0) & (intLX2 < SIZE_3(feature1)) & (intTY2 < SIZE_2(feature1))) {
                float fltLoss = VALUE_4(gradLoss, intN, intC, intY, intX);
                for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                    if ((intLX1 >= 0) & (intTY1 >= 0) & (intRX2 < SIZE_3(feature1)) & (intBY2 < SIZE_2(feature1))) {
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 0, intY, intX)], VALUE_4(feature1, intN, intChannel, intTY1, intLX1) * VALUE_4(feature2, intN, intChannel, intBY2, intRX2) * fltnwx * fltLoss);
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 1, intY, intX)], VALUE_4(feature1, intN, intChannel, intTY1, intLX1) * VALUE_4(feature2, intN, intChannel, intBY2, intRX2) * fltnwy * fltLoss);
                    }
                    if ((intLX1 >= 0) & (intBY1 < SIZE_2(feature1)) & (intRX2 < SIZE_3(feature2)) & (intTY2 >= 0)) {
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 0, intY, intX)], VALUE_4(feature1, intN, intChannel, intBY1, intLX1) * VALUE_4(feature2, intN, intChannel, intTY2, intRX2) * fltswx * fltLoss);
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 1, intY, intX)], VALUE_4(feature1, intN, intChannel, intBY1, intLX1) * VALUE_4(feature2, intN, intChannel, intTY2, intRX2) * fltswy * fltLoss);
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intTY1 >= 0) & (intLX2 >= 0) & (intBY2 < SIZE_2(feature1))) {
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 0, intY, intX)], VALUE_4(feature1, intN, intChannel, intTY1, intRX1) * VALUE_4(feature2, intN, intChannel, intBY2, intLX2) * fltnex * fltLoss);
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 1, intY, intX)], VALUE_4(feature1, intN, intChannel, intTY1, intRX1) * VALUE_4(feature2, intN, intChannel, intBY2, intLX2) * fltney * fltLoss);
                    }
                    if ((intRX1 < SIZE_3(feature1)) & (intBY1 < SIZE_2(feature1)) & (intLX2 >= 0) & (intTY2 >= 0)) {
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 0, intY, intX)], VALUE_4(feature1, intN, intChannel, intBY1, intRX1) * VALUE_4(feature2, intN, intChannel, intTY2, intLX2) * fltsex * fltLoss);
                        atomicAdd(&gradFlow[OFFSET_4(gradFlow, intN, 1, intY, intX)], VALUE_4(feature1, intN, intChannel, intBY1, intRX1) * VALUE_4(feature2, intN, intChannel, intTY2, intLX2) * fltsey * fltLoss);
                    }
                }
            }
        }
    } }
'''

def cupy_kernel(strFunction, intWindowSize, objectVariables):
    strKernel = globals()[strFunction]

    strKernel = strKernel.replace('F_SIZE_H', str((intWindowSize - 1) // 2))
    strKernel = strKernel.replace('F_SIZE', str(intWindowSize))

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')


    return strKernel


@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


class bilateralcorrelation_nn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature1, feature2, SBM, time, md=2):
        ctx.save_for_backward(feature1, feature2, SBM, time)
        ctx.md = md

        intInputBatch, _, intInputHeight, intInputWidth = SBM.size() # (+) feature1 -> SBM
        intInputChannel = feature1.size(1) # (+) intInputChannel is a channel size of the feature map
        intWindowSize = (2 * ctx.md + 1)

        output = feature1.new_zeros(intInputBatch, intWindowSize ** 2, intInputHeight, intInputWidth)

        assert feature1.size() == feature2.size()
        assert SBM.size(1) == 2
        assert (feature1.is_contiguous() == True)
        assert (feature2.is_contiguous() == True)
        assert (SBM.is_contiguous() == True)
        assert (time.is_contiguous() == True)
        assert feature1.device == feature2.device and feature1.device == SBM.device and feature1.device == time.device

        if feature1.is_cuda == True and feature2.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('correlation_forward',
                        cupy_kernel('correlation_forward', intWindowSize, {
                            'feature1': feature1,
                            'feature2': feature2,
                            'flow': SBM,
                            'time': time,
                            'output': output
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, feature1.data_ptr(), feature2.data_ptr(), SBM.data_ptr(), time.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        # end
        return output / torch.sqrt(torch.tensor(intInputChannel).float())


    @staticmethod
    def backward(ctx, gradOutput):
        feature1, feature2, SBM, time = ctx.saved_tensors

        # intInputBatch, _, intInputHeight, intInputWidth = SBM.size() # (+) feature1 -> SBM
        intInputChannel = feature1.size(1) # (+) intInputChannel is a channel size of the feature map
        intWindowSize = (2 * ctx.md + 1)

        gradInput1 = feature1.new_zeros(feature1.size()) if \
            ctx.needs_input_grad[0] == True else None
        gradInput2 = feature2.new_zeros(feature2.size()) if \
            ctx.needs_input_grad[1] == True else None
        gradFlow = SBM.new_zeros(SBM.size()) if \
            ctx.needs_input_grad[2] == True else None

        gradOutput = gradOutput / torch.sqrt(torch.tensor(intInputChannel).float())

        if feature1.is_cuda == True and feature2.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # weight grad
            n = gradOutput.nelement()
            cupy_launch('correlation_backward_feature',
                        cupy_kernel('correlation_backward_feature', intWindowSize, {
                            'gradLoss': gradOutput,
                            'feature1': feature1,
                            'feature2': feature2,
                            'flow': SBM,
                            'time': time,
                            'gradInput1': gradInput1,
                            'gradInput2': gradInput2
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), feature1.data_ptr(), feature2.data_ptr(), SBM.data_ptr(), time.data_ptr(),
                      gradInput1.data_ptr(), gradInput2.data_ptr()],
                stream=Stream
            )

        if gradFlow is not None:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            n = gradOutput.nelement()
            cupy_launch('correlation_backward_flow',
                        cupy_kernel('correlation_backward_flow', intWindowSize, {
                            'gradLoss': gradOutput,
                            'feature1': feature1,
                            'feature2': feature2,
                            'flow': SBM,
                            'time': time,
                            'gradFlow': gradFlow
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), feature1.data_ptr(), feature2.data_ptr(), SBM.data_ptr(), time.data_ptr(),
                      gradFlow.data_ptr()],
                stream=Stream
            )

        # end

        return gradInput1, gradInput2, gradFlow, None, None

# end