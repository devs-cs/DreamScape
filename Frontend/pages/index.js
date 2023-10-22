import { MoonIcon } from 'evergreen-ui';
import Arrow from '../components/Arrow';
import HeaderComp from '../components/HeaderComp';
import SideBar from '../components/SideBar';
import styles from '../styles/Home.module.css'
import Dnd from "./../components/Dnd";
import StarryBackground from '../components/StarryBackground';
import Loading from '../components/Loading';
import EegData from '../components/EegData';
import Text from '../components/Text';
import Video from '../components/Video';
import PContext from '../config/context';
import { useContext } from 'react';
import Spinner from '../components/Spinner';

export default function Home(){
  const {stage} = useContext(PContext);

  return <div styles={styles.container}>
    <HeaderComp></HeaderComp>
    <StarryBackground color="lightgrey" zIndex={-1}></StarryBackground>

    <div className='openingBanner'>
      <h1><MoonIcon size={60}></MoonIcon>DreamScape</h1>
      <p>Bring Your Dreams to Life!</p>
      <StarryBackground color="white" zIndex={10}></StarryBackground>
    </div>


    <SideBar title="Step 1:" subtitle="Create Your Dream!" description="Show patient image and use ML model to parse EEG Data">
      <Dnd></Dnd>
    </SideBar>

    {stage>=2&&<SideBar title="Step 2:" subtitle="Generate Images from Data!" description="Use an ML model to generate images corresponding to EEG data. Displays EEG data visualized graphically, the embedding space vector, and the corresponding generated image by the ML model.">
      <EegData></EegData>
    </SideBar>}

    {stage>=3&&<SideBar title="Step 3:" subtitle="Generate Text and Story!" description="Transformer model translates from image to text, before an LLM recreates your story from text. ">
      <Text></Text>
    </SideBar>}

    {stage>=4&&<SideBar title="Step 4:" subtitle="Generate Video!" description="Use text-to-video model to generate a sequence of images representing video. Given proper time, can 
    generate video. ">
      <Video></Video>
    </SideBar>}

    <footer>DreamScape Demo from HackHarvard 2023</footer>
  </div>
}