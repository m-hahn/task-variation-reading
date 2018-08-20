require 'nn'

local DynamicallySelectTable, parent = torch.class('nn.DynamicallySelectTable', 'nn.Module')

function DynamicallySelectTable:__init(indexContainer)
   parent.__init(self)
   self.indexContainer = indexContainer
   self.gradInput = {}
end

function DynamicallySelectTable:updateOutput(input)
   -- handle negative indices
   local index = self.indexContainer[1] < 0 and #input + self.indexContainer[1] + 1 or self.indexContainer[1]
   assert(input[index], "index does not exist in the input table")
   self.output = input[index]

   return self.output
end

local function zeroTableCopy(t1, t2)
   for k, v in pairs(t2) do
      if (torch.type(v) == "table") then
         t1[k] = zeroTableCopy(t1[k] or {}, t2[k])
      else
         if not t1[k] then
            t1[k] = v:clone():zero()
         else
            t1[k]:resizeAs(v)
            t1[k]:zero()
         end
      end
   end
   for k, v in pairs(t1) do
      if not t2[k] then
         t1[k] = nil
      end
   end
   return t1
end

function DynamicallySelectTable:updateGradInput(input, gradOutput)
   -- make gradInput a zeroed copy of input
--   print(self.indexContainer)
--   print(#input)
--   print(#gradOutput)
--   print(#gradInput)
--   print("5068")


   zeroTableCopy(self.gradInput, input)
   -- handle negative indices
   local index = self.indexContainer[1] < 0 and #input + self.indexContainer[1] + 1 or self.indexContainer[1]
   -- copy into gradInput[index] (necessary for variable sized inputs)
   assert(self.gradInput[index])
   nn.utils.recursiveCopy(self.gradInput[index], gradOutput)

--   print(self.indexContainer)
--   print(#input)
--   print(#gradOutput)
--   print(#gradInput)
--   print("5093")
   return self.gradInput
end

function DynamicallySelectTable:type(type, tensorCache)
   self.gradInput = {}
   self.output = {}
   return parent.type(self, type, tensorCache)
end

function DynamicallySelectTable:__tostring__()
  return torch.type(self) .. '(' .. self.indexContainer .. ')'
end

DynamicallySelectTable.clearState = nn.Identity.clearState
