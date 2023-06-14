import cupy
import torch
import re

'''
Cuda version of bilateral correlation computation
'''


correlation_forward = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void correlation_forward(
        const int n,
        const float* feature1,
        const float* feature2,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        float fltOutput = 0.0;
    
        const int intN  = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intC  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY  = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX  = ( intIndex                                                    ) % SIZE_3(output);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;
        
        int intX1 = intX + k;
        int intY1 = intY + l;
        int intX2 = intX - k;
        int intY2 = intY - l;

        if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(output), SIZE_3(output)) & WITHIN_BOUNDS(intX2, intY2, SIZE_2(output), SIZE_3(output))){
            for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                fltOutput += VALUE_4(feature1, intN, intChannel, intY1, intX1) * VALUE_4(feature2, intN, intChannel, intY2, intX2);                
            }
        }

        output[intIndex] = fltOutput;

    } }
'''

correlation_backward_feature = '''
    #define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

    extern "C" __global__ void correlation_backward_feature(
        const int n,
        const float* gradLoss,
        const float* feature1,
        const float* feature2,
        float* gradInput1,
        float* gradInput2
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        
        const int intN  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss) / SIZE_1(gradLoss) ) % SIZE_0(gradLoss);
        const int intC  = ( intIndex / SIZE_3(gradLoss) / SIZE_2(gradLoss)                    ) % SIZE_1(gradLoss);
        const int intY  = ( intIndex / SIZE_3(gradLoss)                                       ) % SIZE_2(gradLoss);
        const int intX  = ( intIndex                                                          ) % SIZE_3(gradLoss);

        int k = (intC % F_SIZE) - F_SIZE_H;
        int l = (intC / F_SIZE) - F_SIZE_H;

        int intX1 = intX + k;
        int intY1 = intY + l;
        int intX2 = intX - k;
        int intY2 = intY - l;

        if (WITHIN_BOUNDS(intX1, intY1, SIZE_2(gradLoss), SIZE_3(gradLoss)) & WITHIN_BOUNDS(intX2, intY2, SIZE_2(gradLoss), SIZE_3(gradLoss))){
            float fltLoss = VALUE_4(gradLoss, intN, intC, intY, intX);
            for (int intChannel = 0; intChannel < SIZE_1(feature1); intChannel += 1) {
                atomicAdd(&gradInput1[OFFSET_4(gradInput1, intN, intChannel, intY1, intX1)], VALUE_4(feature2, intN, intChannel, intY2, intX2) * fltLoss);
                atomicAdd(&gradInput2[OFFSET_4(gradInput2, intN, intChannel, intY2, intX2)], VALUE_4(feature1, intN, intChannel, intY1, intX1) * fltLoss);
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


class bilateralcorrelation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feature1, feature2, md):
        ctx.save_for_backward(feature1, feature2)
        ctx.md = md

        intInputBatch, intInputChannel, intInputHeight, intInputWidth = feature1.size()
        intWindowSize = (2 * ctx.md + 1)

        output = feature1.new_zeros(intInputBatch, intWindowSize ** 2, intInputHeight, intInputWidth)

        assert feature1.size() == feature2.size()
        assert (feature1.is_contiguous() == True) and (feature2.is_contiguous() == True)
        assert feature1.device == feature2.device

        if feature1.is_cuda == True and feature2.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('correlation_forward',
                        cupy_kernel('correlation_forward', intWindowSize, {
                            'feature1': feature1,
                            'feature2': feature2,
                            'output': output
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, feature1.data_ptr(), feature2.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        # end
        return output / torch.sqrt(torch.tensor(intInputChannel).float())


    @staticmethod
    def backward(ctx, gradOutput):
        feature1, feature2 = ctx.saved_tensors

        # intInputBatch, _, intInputHeight, intInputWidth = SBM.size() # (+) feature1 -> SBM
        intInputChannel = feature1.size(1) # (+) intInputChannel is a channel size of the feature map
        intWindowSize = (2 * ctx.md + 1)

        gradInput1 = feature1.new_zeros(feature1.size()) if \
            ctx.needs_input_grad[0] == True else None
        gradInput2 = feature2.new_zeros(feature2.size()) if \
            ctx.needs_input_grad[1] == True else None
        
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
                            'gradInput1': gradInput1,
                            'gradInput2': gradInput2
                        }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, gradOutput.data_ptr(), feature1.data_ptr(), feature2.data_ptr(),
                      gradInput1.data_ptr(), gradInput2.data_ptr()],
                stream=Stream
            )

        # end

        return gradInput1, gradInput2, None, None

# end
