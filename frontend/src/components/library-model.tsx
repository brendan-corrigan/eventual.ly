import React, { useEffect } from "react";
import { Button } from "./ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "./ui/dialog";

import LibraryModelForm from "./library-model-form";
import { FileIcon } from "lucide-react";

export default function LibraryModel() {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <div className="flex w-full gap-2 p-1 items-center cursor-pointer">
          <FileIcon className="w-4 h-4" />
          <p>Library</p>
        </div>
      </DialogTrigger>
      <DialogContent className="space-y-2 w-full">
        <DialogTitle>Library</DialogTitle>
        <LibraryModelForm />
      </DialogContent>
    </Dialog>
  );
}
