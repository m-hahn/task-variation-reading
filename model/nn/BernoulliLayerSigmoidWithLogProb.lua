require 'nn'

local BernoulliLayerSigmoidWithLogProb, Parent = torch.class('nn.BernoulliLayerSigmoidWithLogProb', 'nn.Module')

function BernoulliLayerSigmoidWithLogProb:__init(...)
   Parent.__init(self)
   self.gradInput = torch.CudaTensor(...):zero()
   self.output = {torch.CudaTensor(...),torch.CudaTensor(...),torch.CudaTensor(...)}
   self.sigmoid = nn.Sigmoid():cuda()
end

-- input: tensor of unnormalized attention scores
-- output: table consisting of (1) table of filter values (1 or 0), (2) log probabilities for the choices, (3) the actual attention values (between zero and one)
-- hard-coded that: the first dimension is the batch size, and the second dimension has size 1
function BernoulliLayerSigmoidWithLogProb:updateOutput(input)
--   print("---------")
  -- print("Unnormalized scores")
--   print(input)
   -- attention scores
   self.output[3]:copy(self.sigmoid:forward(input))
--   print("Attention Scores")
  -- print(self.output[3])
   -- decisions
   self.output[1]:copy(self.output[3])
   self.output[1]:apply(function(attention)
         local experiment = torch.uniform()
         if experiment < attention then
           return 1
         else
           return 0
         end
      end)
  -- print("Decisions")
--   print(self.output[1])
   -- log probabilities of the decisions made
   torch.cmul(self.output[2],self.output[3],torch.add(self.output[1],-0.5):mul(2)):add(1):add(-1,self.output[1]):log()
  -- print("Log Probabilities of Decisions")
--   print(self.output[2])
   return self.output
end

-- unless doing analytical minimization of fixations, expects gradients only to come from the second output node (the log probabilities)
-- the gradient of the sigmoid-log-probability (the second output) with regard to the (unnormalized) attention values
function BernoulliLayerSigmoidWithLogProb:updateGradInput(input, gradOutput)
    self.gradInput:copy(self.output[1]):csub(self.output[3])
    self.gradInput:cmul(gradOutput[2])
    if neatQA.ANALYTICAL_MINIMIZATION_OF_FIXATIONS then -- gradOutput[3] also has to be considered
       -- add: [- sigma(x) * (sigma(x)-1) * gradOutput[3]], where the initial Minus reverses the sign of the the (sigma(x)-1) term to (1-sigma(X))
--       print("Backprop at 4927")
  --     print(self.gradInput)
    --   print(gradOutput[3])
      -- print(self.output[3])
       self.gradInput:add(-1,torch.cmul(self.output[3], torch.add(self.output[3],-1)):cmul(gradOutput[3]))
       --print(self.gradInput)
    end
--    print("------------")
--    print("Sigmoid")
--    print(self.output[3])
--    print("INCOMING gradient")
--    print(gradOutput[2])
--    print("Decisions")
--    print(self.output[1])
--    print("GRADIENT")
--    print(self.gradInput)
    return self.gradInput
end


