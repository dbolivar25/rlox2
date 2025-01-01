use crate::runtime::Value;
use chainmap::ChainMap;

pub type Environment = ChainMap<String, Value>;
