import { useTranslation } from "react-i18next";
export default function ChromaDBOptions() {
  const { t } = useTranslation();
  return (
    <div className="w-full h-10 items-center flex">
      <p className="text-sm font-base text-white text-opacity-60">
        There is no configuration needed for ChromaDB.
      </p>
    </div>
  );
}
