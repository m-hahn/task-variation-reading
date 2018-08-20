require 'nn'
-- should actually be called Multinomial
--


-- TODO don't use this is messed up
local MultinoulliLayer, Parent = torch.class('nn.MultinoulliLayer', 'nn.Module')

function MultinoulliLayer:__init(...)
assert(false, "get the version from lm. this one here is messed up.")
crash()
   Parent.__init(self)
   self.gradInput = torch.CudaTensor(...):zero()
   self.output = {torch.CudaTensor(...),torch.CudaTensor(...)}
end

-- input: tensor of attention scores
-- output: table consisting of (1) table of filter values (1 or 0), (2) probabilities for the choices, (3) numbers of attended items per item (0 <= ... <= 1) , (4) avg entropy
-- hard-coded that: the first dimension is the batch size, and the second dimension has size 1
function MultinoulliLayer:updateOutput(input)
   input = input:view(input:size(1),-1)
   local filterIndices = torch.multinomial(input, 1)

   self.output[1] = self.output[1]:view(input:size(1),-1)
   self.output[2] = self.output[2]:view(input:size(1),-1)

   self.output[1]:zero()
print(self.output)
print(filterIndices)
print(input)
   self.output[1]:scatter(2 , filterIndices, 1)
   self.output[1] = self.output[1]:view(input:size(1),1,-1)


  
--local probabilities = torch.cmul(input, torch.add(filterValues, -0.5):mul(2)):add(torch.mul(filterValues, -1):add(1))]]


   --The probability of the choice made in the sampling of the filterValues is given by:
   --
   --       (1-filterValues) + 2 * (filterValues - 0.5) * input
   --
   -- which results in
   -- (1-1) + 2 * (1-0.5) * attention = attention if the word is attended to
   -- (1-0) + 2 * (0-0.5) * attention = 1 - attention if it is not attended to

-- the probabilities
   torch.cmul(self.output[2],input,torch.add(self.output[1],-0.5):mul(2)):add(1):add(-1,self.output[1])


   return self.output
end

-- the gradient of the probability (the second output) with regard to the attention values

function MultinoulliLayer:updateGradInput(input, gradOutput)
--    return gradOutput[2]
    self.gradInput:copy(self.output[1])
    (self.gradInput):apply(function(x)
       if x == 1 then
          return(1)
       elseif x == 0 then
          return(-1)
       else
          assert(false,x.."")
       end
      end)
    self.gradInput:cmul(gradOutput[2])
    return self.gradInput
--    self.gradInput:resizeAs(input)
  --  self.gradInput:zero()
    --return self.gradInput
--   return input:clone():zero()
end


