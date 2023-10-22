import { Heading, MoonIcon } from "evergreen-ui";

export default function HeaderComp(){
    return <header>
        <Heading fontSize={20} display="flex" alignItems="center"><MoonIcon size={20} marginRight="10px"></MoonIcon>DreamScape </Heading>
    </header>
}