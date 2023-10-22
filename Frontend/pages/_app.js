import { useEffect, useState } from 'react'
import '../styles/globals.css'
import PContext from '../config/context';

function MyApp({ Component, pageProps }) {

  useEffect(()=>{
    var dist = window.innerHeight;
    console.log(dist);
    var headerHeight = 70;
    setScrollPositions([headerHeight+dist,headerHeight+2*dist,headerHeight+3*dist]);
  },[])

  // set global state, aka context
  const [nums,setNums] = useState([]);
  const [scrollPositions,setScrollPositions] = useState([]); //three, one for each transition
  const [stage,setStage] = useState(1);//currently on this stage
  const [texts,setTexts] = useState([]); 
  const [story,setStory] = useState("");
  const contextObj = {nums,setNums,stage,setStage,texts,setTexts,scrollPositions,setStory,story};

  useEffect(()=>{
    if(stage>1&&scrollPositions.length>0) window.scrollTo({top: scrollPositions[stage-2], behavior: "smooth"});
  },[stage])

  return <PContext.Provider value={contextObj}>
    <Component {...pageProps} />
  </PContext.Provider>
}

export default MyApp
